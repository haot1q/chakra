#pragma once

#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "et_feeder_node.h"
#include "protoio.hh"

namespace Chakra {
struct CompareNodes : public std::binary_function<
                          std::shared_ptr<ETFeederNode>,
                          std::shared_ptr<ETFeederNode>,
                          bool> {
  bool operator()(
      const std::shared_ptr<ETFeederNode> lhs,
      const std::shared_ptr<ETFeederNode> rhs) const {
    return lhs->getChakraNode()->id() > rhs->getChakraNode()->id();
  }
};

class ETFeeder {
 public:
  ETFeeder(std::string filename);
  ~ETFeeder();

  void addNode(std::shared_ptr<ETFeederNode> node);
  // Only completed nodes may be removed. Unknown IDs are rejected.
  void removeNode(uint64_t node_id);
  bool hasNodesToIssue();
  // Taking a ready node leases it until completion or an explicit push-back.
  std::shared_ptr<ETFeederNode> getNextIssuableNode();
  void pushBackIssuableNode(uint64_t node_id);
  std::shared_ptr<ETFeederNode> lookupNode(uint64_t node_id);
  // Mark an issued node completed before exposing its children. Exactly once.
  void freeChildrenNodes(uint64_t node_id);
  void readGlobalMetadata();
  std::shared_ptr<ETFeederNode> readNode();
  void readNextWindow();
  void resolveDep();
  void printGraph();
  const std::string& tierManifestDigest() const;
  bool hasTierManifestDigest() const;
  const std::string& serviceBindingDigest() const { return service_binding_digest_; }
  const std::string& serviceActivationId() const { return service_activation_id_; }
  std::optional<uint32_t> serviceRank() const { return service_rank_; }

 private:
  void enqueueIfReady(const std::shared_ptr<ETFeederNode>& node);
  void discardCompletedHistory();
  ProtoInputStream trace_;
  const uint32_t window_size_;
  bool et_complete_;
  std::string tier_manifest_digest_;
  std::string service_binding_digest_;
  std::string service_activation_id_;
  std::optional<uint32_t> service_rank_;

  std::map<uint64_t, std::shared_ptr<ETFeederNode>> dep_graph_{};
  std::unordered_set<uint64_t> dep_free_node_id_set_{};
  std::unordered_set<uint64_t> issued_node_ids_{};
  // Until EOF, retain IDs (not nodes) for children read after parent retirement.
  // At EOF only completed nodes still awaiting removeNode remain in this set.
  std::unordered_set<uint64_t> completed_node_ids_{};
  std::priority_queue<
      std::shared_ptr<ETFeederNode>,
      std::vector<std::shared_ptr<ETFeederNode>>,
      CompareNodes>
      dep_free_node_queue_{};
  std::unordered_set<std::shared_ptr<ETFeederNode>> dep_unresolved_node_set_{};
};

} // namespace Chakra
