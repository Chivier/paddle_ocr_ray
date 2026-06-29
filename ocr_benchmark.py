import requests
from datasets import load_dataset
import time
import random
import base64
import io
import argparse
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import track, Progress
from rich.table import Table
from rich.console import Console
import pandas as pd
from datetime import datetime

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="OCR Benchmark Test with Online/Offline Requests")
    parser.add_argument("-n", "--num_requests", type=int, default=100,
                        help="Number of requests to send")
    parser.add_argument("-t", "--total_time", type=float, default=3.0,
                        help="Total time window to send all requests (seconds)")
    parser.add_argument("-w", "--workers", type=int, default=50,
                        help="Number of worker threads")
    parser.add_argument("-r", "--ratio", type=float, default=0.7,
                        help="Ratio of online requests (0-1)")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/ocr",
                        help="OCR service URL")
    parser.add_argument("--max_images", type=int, default=None,
                        help="Maximum number of images to test")
    parser.add_argument("--output", type=str, default="ocr_benchmark_results.json",
                        help="Output file for results")
    return parser.parse_args()

class OCRBenchmark:
    def __init__(self, args):
        self.args = args
        self.ocr_url = args.url
        self.results = []
        self.start_time = None

    def prepare_dataset(self):
        """加载并准备数据集"""
        console.print("[bold blue]Loading OCR benchmark dataset...[/bold blue]")
        ds = load_dataset("getomni-ai/ocr-benchmark", split="test")

        image_data = []
        failed = []

        # Use num_requests if max_images is not specified
        if self.args.max_images is None:
            self.args.max_images = self.args.num_requests
        
        max_images = self.args.max_images or len(ds)
        ds_subset = list(ds)[:max_images]

        with Progress() as progress:
            task = progress.add_task("[green]Processing images...", total=len(ds_subset))

            for idx, sample in enumerate(ds_subset):
                try:
                    img_byte_arr = io.BytesIO()
                    sample["image"].save(img_byte_arr, format="PNG")
                    img_byte_arr.seek(0)

                    # 为每个请求生成随机时间戳和类型
                    timestamp = random.uniform(0, self.args.total_time * 1000)  # 转换为毫秒
                    is_online = random.random() < self.args.ratio

                    image_data.append({
                        "index": idx,
                        "image_bytes": img_byte_arr,
                        "timestamp": timestamp,
                        "is_online": is_online,
                        "request_type": "online" if is_online else "offline"
                    })

                except Exception as e:
                    console.print(f"[red]Error processing image {idx}: {e}[/red]")
                    failed.append(idx)

                progress.update(task, advance=1)

        # 按时间戳排序
        image_data.sort(key=lambda x: x["timestamp"])

        console.print(f"[green]Successfully processed {len(image_data)} images[/green]")
        console.print(f"[yellow]Failed images: {len(failed)}[/yellow]")
        console.print(f"[blue]Online requests: {sum(1 for x in image_data if x['is_online'])}[/blue]")
        console.print(f"[blue]Offline requests: {sum(1 for x in image_data if not x['is_online'])}[/blue]")

        return image_data

    def submit_ocr_request(self, request_data):
        """提交单个OCR请求"""
        image_base64 = base64.b64encode(request_data["image_bytes"].getvalue()).decode("utf-8")
        payload = {
            "image": image_base64  # Ray Serve expects "image" field
        }

        submit_time = time.time()

        try:
            response = requests.post(self.ocr_url, json=payload, timeout=30)
            response_time = time.time()

            if response.status_code == 200:
                result = response.json()
                result.update({
                    "index": request_data["index"],
                    "submit_time": submit_time,
                    "response_time": response_time,
                    "latency": (response_time - submit_time) * 1000,  # 转换为毫秒
                    "is_online": request_data["is_online"],
                    "request_type": request_data["request_type"],
                    "scheduled_timestamp": request_data["timestamp"],
                    "actual_timestamp": (submit_time - self.start_time) * 1000,
                    "status": "success"
                })
                return result
            else:
                return {
                    "index": request_data["index"],
                    "status": "failed",
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "is_online": request_data["is_online"],
                    "request_type": request_data["request_type"]
                }

        except Exception as e:
            return {
                "index": request_data["index"],
                "status": "error",
                "error": str(e),
                "is_online": request_data["is_online"],
                "request_type": request_data["request_type"]
            }

    def run_benchmark(self, image_data):
        """运行基准测试"""
        console.print(f"\n[bold cyan]Starting benchmark with {len(image_data)} requests...[/bold cyan]")
        console.print(f"[cyan]Total time window: {self.args.total_time} seconds[/cyan]")
        console.print(f"[cyan]Worker threads: {self.args.workers}[/cyan]")

        self.start_time = time.time()
        results = []

        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = []

            with Progress() as progress:
                submit_task = progress.add_task(
                    "[yellow]Submitting requests...",
                    total=len(image_data)
                )

                for request_data in image_data:
                    future = executor.submit(self.submit_ocr_request, request_data)
                    futures.append(future)
                    progress.update(submit_task, advance=1)

                # 收集结果
                process_task = progress.add_task(
                    "[green]Processing responses...",
                    total=len(futures)
                )

                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    progress.update(process_task, advance=1)

        end_time = time.time()
        total_duration = end_time - self.start_time

        self.results = results
        console.print(f"\n[green]Benchmark completed in {total_duration:.2f} seconds[/green]")

        return results

    def analyze_results(self):
        """分析测试结果"""
        df = pd.DataFrame(self.results)

        # 分离成功和失败的请求
        success_df = df[df['status'] == 'success']
        failed_df = df[df['status'] != 'success']

        # 分离online和offline请求
        online_df = success_df[success_df['is_online'] == True]
        offline_df = success_df[success_df['is_online'] == False]

        # 创建结果表格
        table = Table(title="OCR Benchmark Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # 总体统计
        table.add_row("Total Requests", str(len(df)))
        table.add_row("Successful Requests", str(len(success_df)))
        table.add_row("Failed Requests", str(len(failed_df)))
        table.add_row("Success Rate", f"{len(success_df)/len(df)*100:.2f}%")
        table.add_row("", "")

        # Online请求统计
        if len(online_df) > 0:
            online_latencies = online_df['latency'].values
            p50 = np.percentile(online_latencies, 50)
            p90 = np.percentile(online_latencies, 90)
            p99 = np.percentile(online_latencies, 99)

            table.add_row("Online Requests", str(len(online_df)))
            table.add_row("Online P50 Latency", f"{p50:.2f} ms")
            table.add_row("Online P90 Latency", f"{p90:.2f} ms")
            table.add_row("Online P99 Latency", f"{p99:.2f} ms")
            table.add_row("Online Mean Latency", f"{online_latencies.mean():.2f} ms")
            table.add_row("", "")

        # Offline请求统计
        if len(offline_df) > 0:
            total_duration = (self.results[-1]['response_time'] - self.start_time)
            offline_throughput = len(offline_df) / total_duration

            table.add_row("Offline Requests", str(len(offline_df)))
            table.add_row("Offline Throughput", f"{offline_throughput:.2f} req/s")
            table.add_row("Offline Mean Latency", f"{offline_df['latency'].mean():.2f} ms")

        console.print(table)

        # 保存详细结果
        results_summary = {
            "config": vars(self.args),
            "summary": {
                "total_requests": len(df),
                "successful_requests": len(success_df),
                "failed_requests": len(failed_df),
                "success_rate": len(success_df) / len(df) * 100,
                "online": {
                    "count": len(online_df),
                    "p50_latency_ms": float(p50) if len(online_df) > 0 else None,
                    "p90_latency_ms": float(p90) if len(online_df) > 0 else None,
                    "p99_latency_ms": float(p99) if len(online_df) > 0 else None,
                    "mean_latency_ms": float(online_latencies.mean()) if len(online_df) > 0 else None
                },
                "offline": {
                    "count": len(offline_df),
                    "throughput_rps": float(offline_throughput) if len(offline_df) > 0 else None,
                    "mean_latency_ms": float(offline_df['latency'].mean()) if len(offline_df) > 0 else None
                }
            },
            "timestamp": datetime.now().isoformat(),
            "detailed_results": self.results
        }

        # 保存到文件
        with open(self.args.output, 'w') as f:
            json.dump(results_summary, f, indent=2)

        console.print(f"\n[green]Detailed results saved to {self.args.output}[/green]")

        # 保存CSV格式的时间戳数据
        timestamp_df = pd.DataFrame(self.results)
        timestamp_df.to_csv("ocr_benchmark_timestamps.csv", index=False)
        console.print(f"[green]Timestamp data saved to ocr_benchmark_timestamps.csv[/green]")

def main():
    args = parse_args()
    benchmark = OCRBenchmark(args)

    # 准备数据
    image_data = benchmark.prepare_dataset()

    # 运行测试
    results = benchmark.run_benchmark(image_data)

    # 分析结果
    benchmark.analyze_results()

if __name__ == "__main__":
    main()
