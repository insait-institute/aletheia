import argparse
import logging
import os
import pickle
from pathlib import Path

from datasets import Dataset, load_dataset

from utils import run_inference

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

NUM_WORKERS = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()

SP_NODEJS = "You are an expert NodeJS programmer. You will be given a question (problem specification) and will generate a correct JavaScript program that matches the specification and passes all tests. Read the inputs from STDIN (e.g., using 'process.stdin' or the 'readline' module) solve the problem and write the answer to STDOUT (e.g., using 'console.log()'). Do not directly test on the sample inputs. Enclose your code within a JavaScript markdown block. Ensure that when the JavaScript program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."  # noqa: E501

SP_RUST = "You are an expert Rust programmer. You will be given a question (problem specification) and will generate a correct Rust program with a main function (fn main) that matches the specification and passes all tests. Read the inputs from STDIN (e.g., using std::io::stdin()) solve the problem and write the answer to STDOUT (e.g., using println!()). Do not directly test on the sample inputs. Enclose your code within a Rust markdown block. Ensure that when the Rust program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."  # noqa: E501

SP_GO = "You are an expert Go programmer. You will be given a question (problem specification) and will generate a correct Go program with a 'package main' and 'func main()' that matches the specification and passes all tests. Read the inputs from STDIN (e.g., using 'fmt.Scan' or 'bufio.NewScanner(os.Stdin)') solve the problem and write the answer to STDOUT (e.g., using 'fmt.Println()'). Do not directly test on the sample inputs. Enclose your code within a Go markdown block. Ensure that when the Go program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."  # noqa: E501

SP_RUBY = "You are an expert Ruby programmer. You will be given a question (problem specification) and will generate a correct Ruby program that matches the specification and passes all tests. Read the inputs from STDIN (e.g., using 'gets') solve the problem and write the answer to STDOUT (e.g., using 'puts'). Do not directly test on the sample inputs. Enclose your code within a Ruby markdown block. Ensure that when the Ruby program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."  # noqa: E501

SP_SCALA = "You are an expert Scala programmer. You will be given a question (problem specification) and will generate a correct Scala program that matches the specification and passes all tests. Your program should include an 'object' (e.t., 'object Main') with a 'def main(args: Array[String]): Unit' method. Read the inputs from STDIN (e.g., using 'scala.io.StdIn.readLine()' or 'scala.io.StdIn.readInt()') solve the problem and write the answer to STDOUT (e.g., using 'println()'). Do not directly test on the sample inputs. Enclose your code within a Scala markdown block. Ensure that when the Scala program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."  # noqa: E501

SP_CSHARP = """You are an expert C# programmer. You will be given a question (problem specification) and will generate a correct C# program that matches the specification and passes all tests. Your program should include a class (e.g., 'Program') with a 'public static void Main(string[] args)' method. Read the inputs from STDIN (e.g., using 'Console.ReadLine()' or 'Console.In'). Your program must include robust input handling to pass all test cases. This means you should:
1. Always check for null or empty strings returned from Console.ReadLine().
2. Carefully consider the input format to avoid TLE errors from waiting for input that isn't coming.
3. Use safe parsing methods like int.TryParse or double.TryParse within your logic to avoid errors from invalid input.
After reading and parsing the input, solve the problem and write the answer to STDOUT (e.g., using 'Console.WriteLine()' or 'Console.Out'). Your output to STDOUT should only be the final answer; do not print error messages or other text, as this will cause test failures. Do not directly test on the sample inputs. Enclose your code within a C# markdown block. Ensure that when the C# program runs, it reads the inputs, runs the algorithm and writes only the final answer to STDOUT."""  # noqa: E501

SP_D = "You are an expert D programmer. You will be given a question (problem specification) and will generate a correct D program that matches the specification and passes all tests. Your program should include a 'void main(string[] args)' function. Read the inputs from STDIN (e.g., using 'readln()' or 'readf()' from 'std.stdio') solve the problem and write the answer to STDOUT (e.g., using 'writeln()' or 'writef()' from 'std.stdio'). Do not directly test on the sample inputs. Enclose your code within a D markdown block. Ensure that when the D program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."  # noqa: E501


def generate_completions(args):
    data = load_dataset("wetsoledrysoul/Heldout-Set", split="full")
    data = data.select_columns(["prompt_id", "query"])
    data = data.to_pandas()
    data = data.drop_duplicates(subset=["prompt_id", "query"], keep="first").reset_index(drop=True)
    data = Dataset.from_pandas(data)
    log.info(f"Loaded dataset with {len(data)} examples")
    log.info(f"Running inference for {args.model_name}")
    if "gemma-2" in args.model_name:
        prompts = [
            [
                {
                    "role": "user",
                    "content": f"{SP}\n\n{question}",
                }
            ]
            for SP in [SP_CSHARP, SP_D]
            for question in data["query"]
        ]
    else:
        prompts = [
            [
                {
                    "role": "system",
                    "content": SP,
                },
                {"role": "user", "content": question},
            ]
            for SP in [SP_CSHARP, SP_D]
            for question in data["query"]
        ]
    kwargs = {
        "temperature": 1.0,
        "top_p": 0.95,
    }

    completions = run_inference(
        prompts,
        args.model_name,
        tp_size=1,
        n=100,
        max_tokens=2048,
        **kwargs,
    )
    completions = [[nth_response.text for nth_response in responses.outputs] for responses in completions]
    for i, language in enumerate(["csharp", "d"]):
        (args.output_dir / language).mkdir(parents=True, exist_ok=True)
        completions_lang = completions[len(data) * i : len(data) * (i + 1)]
        completions_lang = {idx: completion for idx, completion in zip(data["prompt_id"], completions_lang)}
        with open(args.output_dir / language / "completions.pkl", "wb") as f:
            pickle.dump(completions_lang, f)
        print(f"Saved completions to {args.output_dir / language / 'completions.pkl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Model to use for generating completions")
    args = parser.parse_args()
    save_name = args.model_name.split("/")[-1].replace(".", "_")

    args.output_dir = Path(os.getenv("WORK")) / "rq3" / save_name

    generate_completions(args)
    log.info("Completed")
