import concurrent.futures
import json
import statistics
import time
import urllib.request


URL = "http://127.0.0.1:8000/predict"
TOTAL_REQUESTS = 50
CONCURRENT_REQUESTS = 50

PAYLOAD = json.dumps({
    "features": [0.0] * 53
}).encode("utf-8")


def send_request(request_number):
    start_time = time.perf_counter()

    try:
        request = urllib.request.Request(
            URL,
            data=PAYLOAD,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()
            status_code = response.status

        latency = time.perf_counter() - start_time

        return {
            "request": request_number,
            "success": status_code == 200,
            "latency": latency,
        }

    except Exception as error:
        latency = time.perf_counter() - start_time

        return {
            "request": request_number,
            "success": False,
            "latency": latency,
            "error": str(error),
        }


def main():
    print("=" * 60)
    print("UrbanClap API - Load Test")
    print("=" * 60)
    print(f"Total requests      : {TOTAL_REQUESTS}")
    print(f"Concurrent requests : {CONCURRENT_REQUESTS}")
    print(f"Target              : {URL}")
    print()

    test_start = time.perf_counter()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=CONCURRENT_REQUESTS
    ) as executor:

        futures = [
            executor.submit(send_request, i)
            for i in range(1, TOTAL_REQUESTS + 1)
        ]

        results = [
            future.result()
            for future in concurrent.futures.as_completed(futures)
        ]

    total_time = time.perf_counter() - test_start

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = [r["latency"] for r in results]

    print("-" * 60)
    print("RESULTS")
    print("-" * 60)

    print(f"Total requests      : {len(results)}")
    print(f"Successful requests : {len(successful)}")
    print(f"Failed requests     : {len(failed)}")
    print(f"Total test time     : {total_time:.4f} seconds")

    if latencies:
        print(f"Average latency     : {statistics.mean(latencies) * 1000:.2f} ms")
        print(f"Minimum latency     : {min(latencies) * 1000:.2f} ms")
        print(f"Maximum latency     : {max(latencies) * 1000:.2f} ms")

    if total_time > 0:
        throughput = len(results) / total_time
        print(f"Throughput          : {throughput:.2f} requests/second")

    if failed:
        print()
        print("Errors:")
        for result in failed[:5]:
            print(f"Request {result['request']}: {result.get('error')}")

    print("=" * 60)


if __name__ == "__main__":
    main()