(.venv) PS C:\resilience_bot> python -m app.test_pipeline_client
[WARN] transformers/torch unavailable; HFPipelineLocalClient will operate in stub mode.

===== PIPELINE CLIENT TEST =====
- [stubbed-response] Task: Write a troubleshooting checklist for a reliability engineering question.
- Return exactly 5 short numbered items.
- Each item must be practical and action-oriented.
- Focus on API failures, 504 timeouts, retries, retry storms, outages, upstream dependencies, recent deployments, configuration changes, logs, metrics, and traces.
- Do not write headings. Do not write explanations. Do not write paragraphs.
(.venv) PS C:\resilience_bot> python -m app.main "My API has intermittent 504 timeouts. Give a troubleshooting checklist."  
[WARN] transformers/torch unavailable; HFPipelineLocalClient will operate in stub mode.
[INFO] Attempt 1/3 succeeded in 3 ms

ResilienceBot:

- [stubbed-response] Task: Write a troubleshooting checklist for a reliability engineering question.
- Return exactly 5 short numbered items.
- Each item must be practical and action-oriented.
- Focus on API failures, 504 timeouts, retries, retry storms, outages, upstream dependencies, recent deployments, configuration changes, logs, metrics, and traces.
- Do not write headings. Do not write explanations. Do not write paragraphs.

---
Success: True
Used fallback: False
Attempts: 1
Latency: 4 ms
(.venv) PS C:\resilience_bot> python -m app.test_safe_wrapper

===== TEST CASE: success =====
[INFO] Attempt 1/3 succeeded in 1 ms
Answer: [OK] Reliable answer for: My API has intermittent 504 timeouts. Give a troubleshooting checklist.
Success: True
Used fallback: False
Attempts: 1
Latency ms: 1
Error type: None
Error message: None

===== TEST CASE: retry_then_success =====
[WARN] Attempt 1/3 failed: RuntimeError: Transient model backend failure
[INFO] Waiting 1.5s before retry...
[INFO] Attempt 2/3 succeeded in 0 ms
Answer: [OK after retry] Reliable answer for: My API has intermittent 504 timeouts. Give a troubleshooting checklist.
Success: True
Used fallback: False
Attempts: 2
Latency ms: 1506
Error type: None
Error message: None

===== TEST CASE: timeout =====
[WARN] Attempt 1/3 timed out
[INFO] Waiting 1.5s before retry...
[WARN] Attempt 2/3 timed out
[INFO] Waiting 3.0s before retry...
[WARN] Attempt 3/3 timed out
Answer: ResilienceBot could not generate a reliable response right now. Please retry or consult the runbook.
Success: False
Used fallback: True
Attempts: 3
Latency ms: 19530
Error type: TimeoutError
Error message: Model call exceeded 5.0 seconds

===== TEST CASE: always_fail =====
[WARN] Attempt 1/3 failed: RuntimeError: Persistent backend failure
[INFO] Waiting 1.5s before retry...
[WARN] Attempt 2/3 failed: RuntimeError: Persistent backend failure
[INFO] Waiting 3.0s before retry...
[WARN] Attempt 3/3 failed: RuntimeError: Persistent backend failure
Answer: ResilienceBot could not generate a reliable response right now. Please retry or consult the runbook.
Success: False
Used fallback: True
Attempts: 3
Latency ms: 4566
Error type: RuntimeError
Error message: Persistent backend failure
(.venv) PS C:\resilience_bot> 
