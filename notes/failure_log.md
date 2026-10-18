# Failure log

Qualitative notes from watching evaluation videos. Used for the failure-analysis section (Phase 4):
classify ~10 failures per method with the categories below, then count them.

**Where the videos are:** `outputs/eval/<name>/<checkpoint>_ep<N>/videos/libero_object_<task>/eval_episode_<i>.mp4`
**Which episodes failed:** last column of `results/pilot_sweep.csv` (`1` = success, `0` = failure, in episode order).

## Failure categories

| Code | Category | What it looks like |
|---|---|---|
| G-MISS | Missed grasp | Gripper closes beside / above the object, nothing is picked up |
| G-SLIP | Dropped object | Object is lifted, then falls before reaching the basket |
| NO-RETRY | Continues without object | After a missed grasp, the arm keeps executing the motion to the basket (open-loop) and never re-attempts |
| RETRY-FAIL | Retries, then gives up | Several grasp attempts after a miss, none succeed; then moves on with an empty gripper or runs out of time |
| COLLIDE | Collision | Arm or gripper hits a neighbouring object, knocking it or the target away |
| WRONG-OBJ | Wrong object | Approaches or picks an object other than the instructed one |
| PLACE | Bad placement | Object reaches the basket area but is released outside / falls off |
| TIMEOUT | Stalls / hesitates | Hovers, jitters or moves slowly until the episode time limit |
| OTHER | Other | Describe |

One failure can have two codes (e.g. `G-MISS + NO-RETRY`); list the first cause first.
For **successes** with a near-miss, put the codes in brackets, e.g. `(COLLIDE)`, so they are not counted as failures.
Only episodes 0–9 of each evaluation have video (LeRobot records the first 10 episodes); episodes 10–19 count in the score but cannot be inspected.

---

## ACT · t1 (sim task 3, bbq sauce) · all 46 demos · pilot

### Checkpoints 20k / 40k / 60k / 80k / 100k (n_action_steps=100, default open-loop) · watched Sun 18 Oct

> ⚠ Settings differ from the SmolVLA section below: this table is ACT with the **open-loop default** (100 actions per chunk executed without looking). The final experiment uses `n_action_steps=10` for both methods; the Phase 4 comparison uses those final settings.

| Ckpt | Episode | Result | Codes | Notes |
|---|---|---|---|---|
|20K | 0| 1 | | The task was executed correctly |
|20K | 1| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|20K | 2| 1 | | The task was executed correctly |
|20K | 3| 1 | | The task was executed correctly |
|20K | 4| 1 | | The task was executed correctly |
|20K | 5| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|20K | 6| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|20K | 7| 1 | | The task was executed correctly |
|20K | 8| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|20K | 9| 1 | | The task was executed correctly |
|40K | 0| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|40K | 1| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|40K | 2| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|40K | 3| 1 | (COLLIDE) | The task was executed correctly with a small collision with the object next to the target one |
|40K | 4| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|40K | 5| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|40K | 6| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|40K | 7| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|40K | 8| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|40K | 9| 1 | (COLLIDE) | The task was executed correctly with a small collision with the object next to the target one that make it fall |
|60K | 0| 0 | G-SLIP + NO-RETRY | Gripper couldn't grab the object correctly and keeps moving to the basket |
|60K | 1| 0 | WRONG-OBJ + COLLIDE | This episode is in between the two error, it didn't approach correctly to the object and collide or try to grab the wrong one |
|60K | 2| 1 | | The task was executed correctly |
|60K | 3| 1 | (COLLIDE) | The task was executed correctly with a small collision with the object next to the target one |
|60K | 4| 0 | WRONG-OBJ + COLLIDE | This episode is in between the two error, it didn't approach correctly to the object and collide or try to grab the wrong one |
|60K | 5| 0 | G-SLIP + NO-RETRY | Gripper drop the object middle way (slip) and keeps moving to the basket |
|60K | 6| 0 | WRONG-OBJ + COLLIDE + NO-RETRY | This episode is in between the two error, it didn't approach correctly to the object and collide or try to grab the wrong one |
|60K | 7| 0 | WRONG-OBJ + COLLIDE + NO-RETRY | This episode is in between the two error, it didn't approach correctly to the object and collide or try to grab the wrong one |
|60K | 8| 0 | WRONG-OBJ + COLLIDE + NO-RETRY | This episode is in between the two error, it didn't approach correctly to the object and collide or try to grab the wrong one |
|60K | 9| 1 | | The task was executed correctly |
|80K | 0| 0 | COLLIDE + G-MISS + NO-RETRY | Gripper misses the object and collide with it |
|80K | 1| 0 | WRONG-OBJ + COLLIDE + NO-RETRY | Gripper approach to an object next to the target one and collide with it |
|80K | 2| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|80K | 3| 1 | (COLLIDE) | The task was executed correctly with a small collision with the object next to the target one |
|80K | 4| 1 | (COLLIDE) | The task was executed correctly with a small collision with the object next to the target one |
|80K | 5| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|80K | 6| 1 | | The task was executed correctly |
|80K | 7| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|80K | 8| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|80K | 9| 1 | (COLLIDE) | The task was executed correctly with a small collision with the object next to the target one that make it fall |
|100K | 0| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|100K | 1| 0 | WRONG-OBJ + COLLIDE + NO-RETRY | Gripper approach to an object next to the target one and collide with it |
|100K | 2| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|100K | 3| 1 | (COLLIDE) | The task was executed correctly with a small collision with the object next to the target one |
|100K | 4| 1 | (COLLIDE) | The task was executed correctly with a small collision with the object next to the target one |
|100K | 5| 0 | G-MISS + NO-RETRY | Gripper misses the object and keeps moving toward the basket area, gripper keeps trying to grasp |
|100K | 6| 1 | | The task was executed correctly |
|100K | 7| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|100K | 8| 0 | G-MISS + COLLIDE + NO-RETRY | Gripper misses the object, collide with object next to and keeps moving toward the basket area, gripper keeps trying to grasp|
|100K | 9| 1 | | The task was executed correctly |



General impressions:
- Main failure point is the grasp: the arm's precision at the grasp, not perception.
- Mostly targets the correct object and the basket; at 60k+ a few approaches drift toward the neighbouring object (coded WRONG-OBJ, uncertain: could also be an imprecise approach + COLLIDE).
- When the grasp succeeds, the rest of the task (transport + place) almost always succeeds.
- After a missed grasp it keeps moving toward the basket and "tries to grab" at the same time, and does not retry (NO-RETRY). Hypothesis: open-loop execution of 100-action chunks (~2/3 of an episode) hides the miss from the policy.
- Sometimes the arm hits the object next to the target (COLLIDE) and then fails.

**Closed-loop follow-up (same checkpoints, same 20 scenes):**
- With `n_action_steps=10/20`, success rises to 80–95% at 20k–100k (20k: 16/20 at 10, 17/20 at 20; 40k: 19/20; 100k: 18/20 at 20). Temporal ensembling: 13/20.
- `n_action_steps=10` videos (20k): the only failures in episodes 0–9 are **PLACE** (released badly at the basket); no grasp misses.
- `n_action_steps=20` videos: one episode drops the object and the arm **goes back and re-grasps it** → ACT retries when it re-plans often.
- Conclusion: **NO-RETRY is a property of open-loop chunk execution (100 actions ≈ 2/3 of an episode), not of ACT.** The apparent "longer training hurts" curve above was also an open-loop artefact (later checkpoints commit harder to each chunk).

Sweep context (20 eps each, open-loop default): 20k 10/20, 40k 3/20, 60k 5/20, 80k 4/20, 100k 5/20.

---

## ACT · t1 (sim task 3, bbq sauce) · 5 demos · Phase 2 smoke test

### Checkpoint 10k (n_action_steps=10), 2 episodes · watched Sun 18 Oct

First grid run (`act_obj_t1_n05`, 10k steps, ~110 epochs over 685 frames). 2-episode pipeline test, written to `outputs/smoke.csv` (not part of `results.csv`; the real 3 × 50-episode evaluation runs separately).

| Ckpt | Episode | Result | Codes | Notes |
|---|---|---|---|---|
| 10K | 0 | 0 | G-MISS | Goes for the correct object (bbq sauce) but fails to pick it up; episode runs to the time limit (27 s/ep) |
| 10K | 1 | 0 | G-MISS | Same: correct target, pick fails, episode runs to the time limit |

Impressions:
- With 5 demos ACT still **identifies and reaches the right object**: the failure is grasp precision, not perception or wrong-task data (confirms the 5-episode split points at the right task).
- Same first failure point as the all-demos pilot (the grasp), so far. Check in Phase 4 whether low-data failures shift toward G-MISS / RETRY-FAIL / TIMEOUT (vs PLACE at all demos), and whether ACT retries after a miss at 5 demos.
- 2 episodes say nothing statistically (CI 0–66%); the score comes from the 150-episode evaluation.

---

## SmolVLA · t1 · all 46 demos · pilot

### Checkpoints 5k / 10k / 15k / 20k (n_action_steps=10) · watched Sun 18 Oct

| Ckpt | Episode | Result | Codes | Notes |
|---|---|---|---|---|
| 5K | 0 | 1 | (G-MISS) | The task was executed correctly but the gripper miss the first try but correctly grab at the second try |
| 5K | 1 | 0 | G-MISS + G-SLIP + RETRY-FAIL | Miss in grabbing the object and slip a few times |
| 5K | 2 | 1 |  | The task was executed correctly |
| 5K | 3 | 0 | G-MISS + RETRY-FAIL | Gripper couldn't grab the object even after a few retries, then move the empty gripper to the basket |
| 5K | 4 | 1 |  | The task was executed correctly |
| 5K | 5 | 1 |  | The task was executed correctly |
| 5K | 6 | 1 |  | The task was executed correctly |
| 5K | 7 | 1 |  | The task was executed correctly |
| 5K | 8 | 1 |  | The task was executed correctly |
| 5K | 9 | 1 |  | The task was executed correctly |
| 10K | 0 | 1 |  | The task was executed correctly |
| 10K | 1 | 1 |  | The task was executed correctly |
| 10K | 2 | 1 |  | The task was executed correctly |
| 10K | 3 | 1 |  | The task was executed correctly |
| 10K | 4 | 0 | G-MISS + RETRY-FAIL | Gripper couldn't grab the object even after a few retries, then move the empty gripper to the basket |
| 10K | 5 | 1 |  | The task was executed correctly |
| 10K | 6 | 1 |  | The task was executed correctly |
| 10K | 7 | 0 | G-MISS + RETRY-FAIL | Gripper couldn't grab the object even after a few retries, then move the empty gripper to the basket |
| 10K | 8 | 1 |  | The task was executed correctly |
| 10K | 9 | 1 |  | The task was executed correctly |
| 15K | 0 | 1 | (G-MISS + G-SLIP) | The task was executed correctly, miss in grabbing the object and slip a few times but retry and complete the task |
| 15K | 1 | 1 |  | The task was executed correctly |
| 15K | 2 | 1 |  | The task was executed correctly |
| 15K | 3 | 0 | G-MISS + G-SLIP + RETRY-FAIL | Miss in grabbing the object and slip and it couldn't complete the task, the gripper move to the target empty |
| 15K | 4 | 0 | G-MISS + RETRY-FAIL | Gripper couldn't grab the object even after a few retries, then move the empty gripper to the basket |
| 15K | 5 | 1 |  | The task was executed correctly |
| 15K | 6 | 1 | (PLACE) | The task was executed correctly but the placement was near the edge of the basket almost failing |
| 15K | 7 | 1 |  | The task was executed correctly |
| 15K | 8 | 1 |  | The task was executed correctly |
| 15K | 9 | 0 | G-MISS + RETRY-FAIL | Gripper couldn't grab the object even after a few retries, then move the empty gripper to the basket |
| 20K | 0 | 1 |  | The task was executed correctly |
| 20K | 1 | 1 |  | The task was executed correctly |
| 20K | 2 | 1 |  | The task was executed correctly |
| 20K | 3 | 1 |  | The task was executed correctly |
| 20K | 4 | 0 | G-MISS + RETRY-FAIL | Gripper couldn't grab the object even after a few retries, then move the empty gripper to the basket |
| 20K | 5 | 1 |  | The task was executed correctly |
| 20K | 6 | 1 | (PLACE) | The task was executed correctly but the placement was near the edge of the basket almost failing |
| 20K | 7 | 1 |  | The task was executed correctly |
| 20K | 8 | 1 |  | The task was executed correctly |
| 20K | 9 | 1 |  | The task was executed correctly |



General impressions:
- Main failure point was the grasp: it misses the object or drops it.
- After a missed grasp it usually **retries** (closed loop, re-plans every 10 steps); several near-misses end in success. Failures are repeated misses (RETRY-FAIL), not a lack of retries.
- It correctly identify the target object on all the videos.
- When the grasp succeeds, the rest of the task (transport + place) almost always succeeds.
- Sometimes the gripper place the object at the edge of the basket.
