# 터널 실험 재개 안내 (2026-11-12 16:19 중단 시점)

터널 구간에서 V2V 통신 성공률이 떨어질 때 PPO-Lag, vanilla PPO, PID를 비교하는 실험이다.
설계와 가정은 `docs/implementation_notes.md` 6절, 사전 점검 결과는 `docs/results/2026-11-10_tunnel_precheck.md`에 있다.

## 1. 현재 상태

| 항목 | 상태 | 위치 |
|---|---|---|
| PPO-Lag seed 42 / 43 / 44 | 완료 (각 3000ep, traceback 0) | `outputs/tunnel/train_lag_s{42,43,44}/ppo_lag.pt` |
| vanilla PPO seed 42 / 43 | 완료 (각 3000ep, traceback 0) | `outputs/tunnel/train_ppo_s{42,43}/ppo_lag.pt` |
| vanilla PPO seed 44 | **미완료, 처음부터 재학습 필요** | 실패한 시도: `outputs/tunnel/partial_ppo_1112/` (사용 불가) |
| 평가 | 35칸 중 1칸 완료 (`pid_t0.9`, 100ep) | `outputs/tunnel/eval/` |
| 요약, 결과 문서, 보고 | 시작 전 | |

체크포인트 md5 앞 8자리: PPO-Lag `9ff7f0a6` / `f19f34d8` / `4ca3fdca`, PPO `a6fce015` / `8c20aee6`.

CARLA와 학습·평가 프로세스는 모두 종료되어 있다. `outputs/`는 git에 올리지 않는다(로컬에만 있음).

학습 설정(6개 실행 공통): 3000 에피소드, 터널 밖 p 0.9, 터널 안 평균 p는 에피소드마다 U(0.1, 0.9),
noise std 0.1 (상관 시간 1 s), 리더 급제동 확률 0.5.

## 2. 재개 방법

저장소 루트에서 한 줄만 실행한다. CARLA 서버는 스크립트가 직접 띄운다.

```bash
setsid nohup bash analysis/tunnel_resume.sh > outputs/tunnel/pipeline_resume.log 2>&1 < /dev/null &
```

스크립트가 하는 일:

1. 포트 2000 CARLA를 새로 띄우고 PPO seed 44를 처음부터 학습한다. `train.log`가 10분간 멈추면
   CARLA를 재시작하고 다시 학습한다(최대 3회).
2. 그동안 포트 2010에서 PID, PPO-Lag 3개, PPO seed 42/43을 평가한다. 이미 100ep이 끝난 칸은 건너뛴다.
3. seed 44 학습이 끝나면 그 체크포인트를 두 서버에 나눠 평가한다.

진행 확인: `outputs/tunnel/pipeline_resume.log`에 `TRAIN_DONE`(또는 `TRAIN_FAILED`), 마지막에
`EVAL_DONE failed_cells=N`이 찍힌다. 칸별 진행은 `outputs/tunnel/eval_2000.log`, `eval_2010.log`.

예상 소요: 약 2시간 45분 (학습 1.5~2시간, 평가는 학습과 겹쳐 진행). CARLA가 멈추면 더 걸린다.

## 3. 지켜야 할 것

- **seed 44 학습이 끝날 때까지 `episode_ppo_v2.py`, `v2v_comm.py`, `configs/scenario_town03.json`을 수정하지 않는다.**
  이 커밋의 세 파일은 끝난 5개 실행을 학습시킨 코드와 같다(각 실행 폴더의 `code.diff` = 945a847 대비 차이).
  바뀌면 PPO-Lag과 PPO를 같은 코드로 비교할 수 없다.
- **CARLA는 2대만 쓴다.** 3대째는 8 GB VRAM에서 `VK_ERROR_DEVICE_LOST`로 죽고, 서버가 로딩 중일 때
  시작한 학습 프로세스는 CUDA out of memory로 죽는다. 서버가 뜬 뒤 1분쯤 기다렸다가 클라이언트를 붙인다.
- 서버 하나에 클라이언트는 하나만 붙인다.
- 학습은 중간 재개가 안 된다(`--ppo-fresh`, 난수 흐름이 달라짐). 멈춘 실행은 버리고 처음부터 돌린다.

## 4. 평가가 끝난 뒤 할 일

1. 칸마다 100ep인지 확인한다 (`failed_cells=0`).
2. 요약: `python3 analysis/tunnel_summarize.py LABEL=outputs/tunnel/eval/<칸> ...` 을 전체,
   `--brake only`, `--brake none` 세 가지로 실행한다.
3. PPO-Lag과 PPO는 seed 3개를 묶어 PID와 비교한다 (터널 p 0.9 / 0.7 / 0.5 / 0.3 / 0.1별).
4. 결과를 `docs/results/`에 문서로 남긴다.

평가 설정: 칸당 100ep, 평가 seed 1000 (모든 제어기가 같은 리더 속도·패킷 손실·급제동을 겪음),
터널 밖 p 0.9, noise std 0.1, 급제동 확률 0.5.

## 5. 2026-11-12에 있었던 문제

| 문제 | 결과 | 조치 |
|---|---|---|
| CARLA 3대 동시 실행 시 2대가 기동 직후 죽음 | 처음엔 seed 42만 학습됨 | 2대 구성으로 변경 |
| seed 44 학습 중 CARLA tick이 멈춤 (ep 1752, 로그 1시간 50분 정지) | 그 실행 폐기 | `tunnel_resume.sh`에 로그 정지 감시 추가 |
| 평가 프로세스를 끄기 전에 같은 서버에 학습을 붙임 | 그 실행 폐기 | 서버 재시작 후 다시 시작 |

측정된 속도: 학습 1개만 돌릴 때 82분, 2개 동시에 약 2시간씩, PID 평가 1칸(100ep) 약 4분.
