#!/bin/bash
# Session 6: Day 1b version of the mixed recipe, then stress test all six in a new folder (stress_test_s6 stays as is)
cd ~/Downloads/MLOE---Group-4
bash results/logs/s6_train.sh > results/logs/s6_train2.log 2>&1
grep -q FAILED results/logs/s6_train2.log && { echo S6_FINAL_FAILED; exit 1; }
sed 's/refine_ft,refine_d1b,refine_mix_full,refine_lsui,refine_lsui_uieb/refine_ft,refine_d1b,refine_mix_full,refine_mix_d1b,refine_lsui,refine_lsui_uieb/; s/stress_test_s6/stress_test_s6b/g; s/S6_STRESS/S6B_STRESS/g' results/logs/s6_stress.sh > results/logs/s6_stress_b.sh
bash results/logs/s6_stress_b.sh > results/logs/s6_stress_b.log 2>&1
tail -1 results/logs/s6_stress_b.log | grep -q S6B_STRESS_DONE && echo S6_FINAL_DONE || echo S6_FINAL_FAILED
