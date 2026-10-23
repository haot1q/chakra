#include "et_feeder.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace std;
using namespace Chakra;

namespace {
void removeDependency(const shared_ptr<ETFeederNode>& node, uint64_t parent_id) {
  auto* dependencies = node->getChakraNode()->mutable_data_deps();
  dependencies->erase(
      remove(dependencies->begin(), dependencies->end(), parent_id),
      dependencies->end());
}
} // namespace

ETFeeder::ETFeeder(string filename)
    : trace_(filename), window_size_(4096 * 256), et_complete_(false) {
  if (!trace_.is_open()) { // Assuming a method to check if file is open
    throw std::runtime_error("Failed to open trace file: " + filename);
  }

  try {
    readGlobalMetadata();
    readNextWindow();
  } catch (const std::exception& e) {
    cerr << "Error in constructor: " << e.what() << endl;
    throw; // Rethrow the exception for caller to handle
  }
}

ETFeeder::~ETFeeder() {}

void ETFeeder::addNode(shared_ptr<ETFeederNode> node) {
  if (node == nullptr || completed_node_ids_.count(node->id()) != 0 ||
      !dep_graph_.emplace(node->id(), node).second) {
    throw invalid_argument("Cannot add a null or duplicate ET node");
  }
}

void ETFeeder::removeNode(uint64_t node_id) {
  if (dep_graph_.count(node_id) == 0) {
    throw out_of_range("Cannot remove unknown ET node " + to_string(node_id));
  }
  if (completed_node_ids_.count(node_id) == 0) {
    throw logic_error("Cannot remove incomplete ET node " + to_string(node_id));
  }
  dep_graph_.erase(node_id);
  if (et_complete_) completed_node_ids_.erase(node_id);

  if (!et_complete_ && (dep_free_node_queue_.size() < window_size_)) {
    readNextWindow();
  }
}

bool ETFeeder::hasNodesToIssue() {
  return !(dep_graph_.empty() && dep_free_node_queue_.empty());
}

shared_ptr<ETFeederNode> ETFeeder::getNextIssuableNode() {
  if (dep_free_node_queue_.size() != 0) {
    shared_ptr<ETFeederNode> node = dep_free_node_queue_.top();
    const auto id = node->id();
    if (completed_node_ids_.count(id) != 0 ||
        dep_free_node_id_set_.erase(id) != 1 ||
        !issued_node_ids_.emplace(id).second) {
      throw logic_error("ET ready queue state mismatch for node " + to_string(id));
    }
    dep_free_node_queue_.pop();
    return node;
  } else {
    return nullptr;
  }
}

void ETFeeder::pushBackIssuableNode(uint64_t node_id) {
  const auto node = lookupNode(node_id);
  if (issued_node_ids_.count(node_id) == 0 ||
      completed_node_ids_.count(node_id) != 0 ||
      node->getChakraNode()->data_deps_size() != 0) {
    throw logic_error("Cannot push back ET node not in flight: " + to_string(node_id));
  }
  issued_node_ids_.erase(node_id);
  enqueueIfReady(node);
}

shared_ptr<ETFeederNode> ETFeeder::lookupNode(uint64_t node_id) {
  try {
    return dep_graph_.at(node_id);
  } catch (const std::out_of_range& e) {
    std::cerr << "looking for node_id=" << node_id
              << " in dep graph, however, not loaded yet" << std::endl;
    throw(e);
  }
}

void ETFeeder::freeChildrenNodes(uint64_t node_id) {
  const auto node = lookupNode(node_id);
  if (issued_node_ids_.count(node_id) == 0 ||
      completed_node_ids_.count(node_id) != 0) {
    throw logic_error("Cannot complete ET node not in flight: " + to_string(node_id));
  }
  completed_node_ids_.emplace(node_id);
  issued_node_ids_.erase(node_id);
  for (auto child : node->getChildren()) {
    removeDependency(child, node_id);
    enqueueIfReady(child);
  }
}

void ETFeeder::enqueueIfReady(const shared_ptr<ETFeederNode>& node) {
  const auto id = node->id();
  if (node->getChakraNode()->data_deps_size() == 0 &&
      issued_node_ids_.count(id) == 0 && completed_node_ids_.count(id) == 0 &&
      dep_free_node_id_set_.emplace(id).second) {
    dep_free_node_queue_.emplace(node);
  }
}

void ETFeeder::discardCompletedHistory() {
  for (auto it = completed_node_ids_.begin(); it != completed_node_ids_.end();) {
    if (dep_graph_.count(*it) == 0) {
      it = completed_node_ids_.erase(it);
    } else {
      ++it;
    }
  }
  completed_node_ids_.rehash(0);
}

void ETFeeder::readGlobalMetadata() {
  if (!trace_.is_open()) {
    throw runtime_error(
        "Trace file closed unexpectedly during reading global metadata.");
  }
  shared_ptr<ChakraProtoMsg::GlobalMetadata> pkt_msg =
      make_shared<ChakraProtoMsg::GlobalMetadata>();
  trace_.read(*pkt_msg);
  for (const auto& attr : pkt_msg->attr()) {
    if (attr.name() == "service_rank") {
      if (service_rank_.has_value() || !attr.has_uint64_val() || attr.uint64_val() > UINT32_MAX) {
        throw runtime_error("Duplicate or invalid service_rank metadata");
      }
      service_rank_ = static_cast<uint32_t>(attr.uint64_val());
      continue;
    }
    std::string* destination = nullptr;
    if (attr.name() == "tier_manifest_digest") destination = &tier_manifest_digest_;
    if (attr.name() == "service_binding_digest") destination = &service_binding_digest_;
    if (attr.name() == "service_activation_id") destination = &service_activation_id_;
    if (destination == nullptr) continue;
    if (!destination->empty()) {
      throw runtime_error("Duplicate " + attr.name() + " metadata attribute");
    }
    if (!attr.has_string_val() || attr.string_val().empty()) {
      throw runtime_error(attr.name() + " metadata must be a nonempty string");
    }
    *destination = attr.string_val();
  }
}

const string& ETFeeder::tierManifestDigest() const {
  return tier_manifest_digest_;
}

bool ETFeeder::hasTierManifestDigest() const {
  return !tier_manifest_digest_.empty();
}

shared_ptr<ETFeederNode> ETFeeder::readNode() {
  shared_ptr<ChakraProtoMsg::Node> pkt_msg =
      make_shared<ChakraProtoMsg::Node>();
  if (!trace_.read(*pkt_msg)) {
    return nullptr;
  }
  shared_ptr<ETFeederNode> node = make_shared<ETFeederNode>(pkt_msg);

  bool dep_unresolved = false;
  auto* dependencies = pkt_msg->mutable_data_deps();
  for (auto it = dependencies->begin(); it != dependencies->end();) {
    // A completed parent may still be resident inside an outer callback.
    if (completed_node_ids_.count(*it) != 0) {
      it = dependencies->erase(it);
      continue;
    }
    auto parent_node = dep_graph_.find(*it);
    if (parent_node != dep_graph_.end()) {
      parent_node->second->addChild(node);
    } else {
      dep_unresolved = true;
      node->addDepUnresolvedParentID(*it);
    }
    ++it;
  }

  if (dep_unresolved) {
    dep_unresolved_node_set_.emplace(node);
  }

  return node;
}

void ETFeeder::resolveDep() {
  for (auto it = dep_unresolved_node_set_.begin();
       it != dep_unresolved_node_set_.end();) {
    shared_ptr<ETFeederNode> node = *it;
    vector<uint64_t> dep_unresolved_parent_ids =
        node->getDepUnresolvedParentIDs();
    for (auto inner_it = dep_unresolved_parent_ids.begin();
         inner_it != dep_unresolved_parent_ids.end();) {
      auto parent_node = dep_graph_.find(*inner_it);
      if (completed_node_ids_.count(*inner_it) != 0) {
        removeDependency(node, *inner_it);
        inner_it = dep_unresolved_parent_ids.erase(inner_it);
      } else if (parent_node != dep_graph_.end()) {
        parent_node->second->addChild(node);
        inner_it = dep_unresolved_parent_ids.erase(inner_it);
      } else {
        ++inner_it;
      }
    }
    if (dep_unresolved_parent_ids.size() == 0) {
      it = dep_unresolved_node_set_.erase(it);
    } else {
      node->setDepUnresolvedParentIDs(dep_unresolved_parent_ids);
      ++it;
    }
  }
}

void ETFeeder::readNextWindow() {
  if (!trace_.is_open()) {
    throw runtime_error(
        "Trace file closed unexpectedly during reading next window.");
  }
  uint32_t num_read = 0;
  do {
    shared_ptr<ETFeederNode> new_node = readNode();
    if (new_node == nullptr) {
      et_complete_ = true;
      discardCompletedHistory();
      break;
    }

    addNode(new_node);
    ++num_read;

    resolveDep();
  } while ((num_read < window_size_) || (dep_unresolved_node_set_.size() != 0));

  for (const auto& node_id_node : dep_graph_) {
    enqueueIfReady(node_id_node.second);
  }
}

void ETFeeder::printGraph() {
  for (auto const &pair: dep_graph_) {
      cout << "{" << pair.first << ": ";
      pair.second->printNode();
  }
}
