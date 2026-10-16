// Unit tests for the block pool, victim selection and simulator invariants (no framework: assert + exit code).
#include <cassert>
#include <cstdio>
#include <vector>

#include "../src/kv_pool.hpp"
#include "../src/sim.hpp"

using namespace kvsim;

static void test_pool() {
  BlockPool p(4, 16);
  std::vector<int> a, b;
  assert(p.blocks_for(1) == 1 && p.blocks_for(16) == 1 && p.blocks_for(17) == 2);
  assert(p.grow_to(a, 33));            // 3 blocks
  assert(p.free_blocks() == 1);
  assert(!p.grow_to(b, 40));           // needs 3, only 1 free: no partial allocation
  assert(b.empty() && p.free_blocks() == 1);
  assert(p.grow_to(a, 64));            // grows in place to 4 blocks
  assert(p.free_blocks() == 0);
  p.release(a);
  assert(p.free_blocks() == 4 && a.empty());
}

static void test_victims() {
  // now = 10. A idle since 1 (returns at 30), B idle since 9 (returns at 10.5).
  std::vector<IdleCandidate> c = {{0, 1.0, 30.0, 0.2}, {1, 9.0, 10.5, 5.0}};
  assert(pick_victim(Policy::Lru, c, 10) == 0);     // longest idle
  assert(pick_victim(Policy::Oracle, c, 10) == 0);  // furthest true return
  assert(pick_victim(Policy::Hint, c, 10) == 1);    // B ran a "slow" tool per the hint, A a fast one
  assert(pick_victim(Policy::Lru, {}, 10) == -1);
}

static void test_sim_invariants() {
  std::vector<Trace> traces = {{{500, 50, 0.01, "calc"}, {560, 40, 0.3, "lookup"}, {620, 30, -1, ""}}};
  Config cfg;
  cfg.sessions = 200;
  cfg.arrival_per_s = 0.2;  // well under saturation (~1.3 sessions/s for this trace)
  cfg.tools.web_s = 1;
  for (Policy p : {Policy::Drop, Policy::Lru, Policy::Hint, Policy::Oracle}) {
    cfg.policy = p;
    Result r = simulate(traces, cfg);
    assert(r.sessions_done == cfg.sessions);
    assert(r.prompt_tokens == 200LL * (500 + 560 + 620));
    assert(r.prefilled_tokens <= r.prompt_tokens);
    if (p == Policy::Drop) assert(r.prefilled_tokens == r.prompt_tokens);
    // With ample memory nothing is evicted: later calls only prefill their new tokens.
    if (p != Policy::Drop) {
      assert(r.evictions == 0);
      assert(r.prefilled_tokens == 200LL * (500 + (560 - 550) + (620 - 600)));
    }
  }
  // Tight memory forces evictions under retention policies.
  cfg.kv_tokens = 4096;
  cfg.arrival_per_s = 20;
  cfg.policy = Policy::Lru;
  Result r = simulate(traces, cfg);
  assert(r.sessions_done == cfg.sessions && r.evictions > 0);
  // Session-aware admission never lets a new session evict, so it recomputes no more than FIFO.
  cfg.admission = Admission::Session;
  Result rs = simulate(traces, cfg);
  assert(rs.sessions_done == cfg.sessions);
  assert(rs.prefilled_tokens <= r.prefilled_tokens);
}

int main() {
  test_pool();
  test_victims();
  test_sim_invariants();
  std::puts("all tests passed");
  return 0;
}
