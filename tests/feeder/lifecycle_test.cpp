#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "et_feeder.h"

namespace {
constexpr uint64_t kWindow = 4096 * 256;
using Chakra::ETFeeder;
using ChakraProtoMsg::COMP_NODE;
using ChakraProtoMsg::MEM_LOAD_NODE;

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void rejects(Operation operation, const std::string& message) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, message);
}

class TraceFile {
 public:
  explicit TraceFile(const std::function<void(ProtoOutputStream&)>& write) {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "chakra-lifecycle-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    const char* directory = mkdtemp(buffer.data());
    if (directory == nullptr) throw std::runtime_error("cannot create test directory");
    directory_ = directory;
    path_ = directory_ / "graph.et";
    ProtoOutputStream output(path_.string());
    ChakraProtoMsg::GlobalMetadata metadata;
    metadata.set_version("feeder-lifecycle-test-v1");
    output.write(metadata);
    write(output);
  }

  ~TraceFile() {
    std::error_code ignored;
    std::filesystem::remove(path_, ignored);
    std::filesystem::remove(directory_, ignored);
  }

  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path directory_;
  std::filesystem::path path_;
};

void writeNode(ProtoOutputStream& output, uint64_t id,
               std::initializer_list<uint64_t> dependencies = {},
               ChakraProtoMsg::NodeType type = MEM_LOAD_NODE) {
  ChakraProtoMsg::Node node;
  node.set_id(id);
  node.set_name("lifecycle");
  node.set_type(type);
  node.set_duration_micros(type == COMP_NODE ? 0 : 1);
  for (const auto parent : dependencies) node.add_data_deps(parent);
  output.write(node);
}

auto take(ETFeeder& feeder, uint64_t expected) {
  auto node = feeder.getNextIssuableNode();
  require(node != nullptr, "missing ready node " + std::to_string(expected));
  require(node->id() == expected,
          "expected ready node " + std::to_string(expected) + ", got " +
              std::to_string(node->id()));
  return node;
}

void complete(ETFeeder& feeder, uint64_t id) {
  feeder.freeChildrenNodes(id);
  feeder.removeNode(id);
}

uint64_t drain(ETFeeder& feeder) {
  uint64_t count = 0;
  while (const auto node = feeder.getNextIssuableNode()) {
    complete(feeder, node->id());
    ++count;
  }
  require(!feeder.hasNodesToIssue(), "graph did not drain completely");
  return count;
}

void smallLifecycle() {
  TraceFile trace([](ProtoOutputStream& output) {
    writeNode(output, 1);
    writeNode(output, 2);
    writeNode(output, 3, {1, 1, 2});
    writeNode(output, 7, {3});
  });
  ETFeeder feeder(trace.path());
  take(feeder, 1);
  take(feeder, 2);
  require(feeder.getNextIssuableNode() == nullptr, "blocked child was issued");
  rejects([&] { feeder.pushBackIssuableNode(3); }, "pushed a blocked node");
  rejects([&] { feeder.pushBackIssuableNode(999); }, "pushed an unknown node");
  rejects([&] { feeder.freeChildrenNodes(999); }, "completed an unknown node");
  rejects([&] { feeder.removeNode(999); }, "removed an unknown node");
  rejects([&] { feeder.removeNode(2); }, "removed an incomplete node");
  feeder.pushBackIssuableNode(1);
  rejects([&] { feeder.pushBackIssuableNode(1); }, "pushed a ready node twice");
  take(feeder, 1);
  feeder.freeChildrenNodes(1);
  rejects([&] { feeder.freeChildrenNodes(1); }, "completed a node twice");
  rejects([&] { feeder.pushBackIssuableNode(1); }, "pushed a completed node");
  require(feeder.lookupNode(3)->getChakraNode()->data_deps_size() == 1,
          "completion did not remove all repeated dependencies");
  require(feeder.getNextIssuableNode() == nullptr, "joined child issued too soon");
  feeder.removeNode(1);
  rejects([&] { feeder.removeNode(1); }, "removed a retired node twice");
  complete(feeder, 2);
  take(feeder, 3);
  complete(feeder, 3);
  take(feeder, 7);
  complete(feeder, 7);
  require(!feeder.hasNodesToIssue(), "small graph retained state");
}

void sparseForwardDependencies() {
  TraceFile trace([](ProtoOutputStream& output) {
    writeNode(output, 4, {900, 900});
    writeNode(output, 900);
  });
  ETFeeder feeder(trace.path());
  take(feeder, 900);
  complete(feeder, 900);
  take(feeder, 4);
  complete(feeder, 4);
  require(!feeder.hasNodesToIssue(), "forward-reference graph retained state");
}

void nestedCompletion(uint64_t count) {
  TraceFile trace([count](ProtoOutputStream& output) {
    writeNode(output, 1);
    writeNode(output, 2, {1}, COMP_NODE);
    writeNode(output, 3, {2});
    // Independent filler roots bound the graph depth during normal teardown.
    for (uint64_t id = 4; id < count; ++id) writeNode(output, id);
    writeNode(output, count, {1});
  });
  ETFeeder feeder(trace.path());
  take(feeder, 1);
  feeder.freeChildrenNodes(1);
  auto zero = take(feeder, 2);
  require(zero->runtime() == 0, "nested child must complete synchronously");
  complete(feeder, 2);
  // The parent's outer callback has not removed node 1 yet.
  take(feeder, 3);
  require(feeder.lookupNode(count)->getChakraNode()->data_deps_size() == 0,
          "late child kept a dependency on a completed resident parent");
  feeder.removeNode(1);
  complete(feeder, 3);
  require(drain(feeder) == count - 3, "nested completion lost or duplicated nodes");
}

void removedParent() {
  TraceFile trace([](ProtoOutputStream& output) {
    writeNode(output, 1);
    for (uint64_t id = 2; id <= kWindow; ++id) writeNode(output, id);
    writeNode(output, kWindow + 1, {1, 1});
  });
  ETFeeder feeder(trace.path());
  take(feeder, 1);
  complete(feeder, 1);
  require(feeder.lookupNode(kWindow + 1)->getChakraNode()->data_deps_size() == 0,
          "late child kept a dependency on a removed parent");
  require(drain(feeder) == kWindow, "late-child graph did not drain exactly once");
}

void concurrentIssuedNodes() {
  TraceFile trace([](ProtoOutputStream& output) {
    for (uint64_t id = 1; id <= kWindow + 1; ++id) writeNode(output, id);
  });
  ETFeeder feeder(trace.path());
  take(feeder, 1);
  const auto second = take(feeder, 2);
  complete(feeder, 1);
  take(feeder, 3);
  require(feeder.lookupNode(2) == second, "refill replaced an in-flight node");
  complete(feeder, 2);
  complete(feeder, 3);
  require(drain(feeder) == kWindow - 2, "concurrent graph duplicated a node");
}
} // namespace

int main(int argc, char** argv) {
  try {
    require(argc == 2, "expected one lifecycle test case name");
    const std::string mode = argv[1];
    if (mode == "small") {
      smallLifecycle();
      sparseForwardDependencies();
      nestedCompletion(8);
    } else if (mode == "window") {
      nestedCompletion(kWindow);
    } else if (mode == "window-plus-one") {
      nestedCompletion(kWindow + 1);
    } else if (mode == "removed-parent") {
      removedParent();
    } else if (mode == "concurrent") {
      concurrentIssuedNodes();
    } else {
      throw std::runtime_error("unknown lifecycle test case: " + mode);
    }
    std::cout << "PASS " << mode << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}
