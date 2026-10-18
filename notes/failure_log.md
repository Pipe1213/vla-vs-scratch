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
| COLLIDE | Collision | Arm or gripper hits a neighbouring object, knocking it or the target away |
| WRONG-OBJ | Wrong object | Approaches or picks an object other than the instructed one |
| PLACE | Bad placement | Object reaches the basket area but is released outside / falls off |
| TIMEOUT | Stalls / hesitates | Hovers, jitters or moves slowly until the episode time limit |
| OTHER | Other | Describe |

One failure can have two codes (e.g. `G-MISS + NO-RETRY`); list the first cause first.

---

## ACT · t1 (sim task 3, bbq sauce) · all 46 demos · pilot

### Checkpoints 20k / 40k / 60k (n_action_steps=100, default) · watched Sun 18 Oct

| Ckpt | Episode | Result | Codes | Notes |
|---|---|---|---|---|
| | | | | |

General impressions:
- Main failure point is the grasp: the arm's precision at the grasp, not perception.
- It identifies the correct object and the basket correctly (no WRONG-OBJ seen).
- When the grasp succeeds, the rest of the task (transport + place) almost always succeeds.
- After a missed grasp it keeps moving toward the basket and "tries to grab" at the same time, and does not retry (NO-RETRY). Hypothesis: open-loop execution of 100-action chunks (~2/3 of an episode) hides the miss from the policy.
- Sometimes the arm hits the object next to the target (COLLIDE) and then fails.

Sweep context (20 eps each): 20k 10/20, 40k 3/20, 60k 5/20, 80k 4/20, 100k 5/20.

---

## SmolVLA · t1 · all 46 demos · pilot

### Checkpoints 5k / 10k / 15k / 20k (n_action_steps=10)

| Ckpt | Episode | Result | Codes | Notes |
|---|---|---|---|---|
| | | | | |

General impressions:
-
