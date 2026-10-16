# fp8 KV 캐시가 Qwen2.5-7B-AWQ 출력을 망가뜨린 원인 분석

RTX 3060 Ti(sm86), vLLM 0.11.2, FlashInfer 0.5.2. 결과 해석은 상위 [README](../../README.md) 실험 3의 "왜 깨졌는지 좁혀 봤습니다"에 있습니다.

| 파일 | 내용 |
|---|---|
| `sanity_runs.txt` | 서버 설정별 확인 질문("What is 17*23?") 응답, 사용한 어텐션 백엔드, Triton 백엔드 실패 원인 |
| `fi_fp8_check.py` | FlashInfer 디코드·prefill 커널: fp8 KV 결과 vs 같은 값을 fp16으로 푼 KV 결과 (커널 오차만 분리) |
| `cache_write_check.py` | vLLM `reshape_and_cache_flash`의 fp8 기록 바이트 vs `torch` 캐스트 |
| `kv_outliers.py` | fp16 KV로 실행 중인 vLLM에 훅을 걸어 실제 레이어별 Q/K/V를 잡고, K/V 저장 방식별 어텐션 출력 오차 계산 |
| `kv_outliers.json` | 위 스크립트의 레이어별 결과 (max\|K\|, max\|V\|, 방식별 오차) |
| `../results_7b_fp8_e4m3scale/` | `--calculate-kv-scales`로 돌린 NAT 벤치 (동시 1/16 × 3회, 정답률 0) |
| `kv_emu/` | vLLM 일반 플러그인. fp16 캐시 위에서 레이어별·값별 e4m3 반올림을 에뮬레이션 (`KV_EMU`, `KV_EMU_KEEP_LAYERS`, `KV_EMU_TOP`) |
| `mixed_bench.sh` | 혼합 정밀도 5개 조건(off, all, keep0_27, keep8, top5)을 NAT 벤치로 실행 |
| `emu_logs/` | 조건별 서버 로그 요약(플러그인 적용 확인 줄)과 벤치 진행 로그 |
| `../results_7b_emu_*/` | 조건별 결과. 정답률(동시 1/16): off 50/56%, all 0/0%, keep0_27 25/50%, keep8 38/44%, top5 6/6% |

```bash
# GPU가 비어 있을 때 (vLLM 서버 종료 후), serve venv에서
python fi_fp8_check.py
python cache_write_check.py
HF_HOME=/data/hf-cache python kv_outliers.py kv_outliers.json

# 혼합 정밀도 (nat_langgraph/에서, vLLM 서버 없이 시작)
uv pip install --python $SERVE_VENV/bin/python -e fp8_kv/kv_emu
./fp8_kv/mixed_bench.sh
```
