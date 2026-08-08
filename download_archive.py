import argparse
from dotenv import load_dotenv
import os
from pathlib import Path
import httpx

load_dotenv()

EVENT_TYPE = "EARNINGS_RELEASE"
ARCHIVE_ROOT = Path("data/archive")

#parameters that you pass to the program when running it from the terminal

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "quarter"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    quarter = args.quarter

    api_key = os.getenv("EM_API_KEY")
    base_url = os.getenv("EM_API_BASE_URL")

    if not api_key:
        raise RuntimeError("EM_API_KEY не задан")

    if not base_url:
        raise RuntimeError("EM_API_BASE_URL не задан")


    with httpx.Client(
        base_url=base_url,
        headers={"X-API-Key": api_key},
        timeout=30.0,
    ) as client:
        response = client.get("/archive")
        response.raise_for_status()
        manifest = response.json()

    matching_files = []

    for item in manifest["files"]:
        if ( item["event_type"] == EVENT_TYPE and item["quarter"] == quarter):
            matching_files.append(item)
  
    if not matching_files:
        raise RuntimeError(
            f"Файл {EVENT_TYPE} / {quarter} не найден"
        )

    archive_info = matching_files[0]

    quarter_directory = ARCHIVE_ROOT / quarter
    quarter_directory.mkdir(parents=True, exist_ok=True,)

    output_path = (
        quarter_directory
        / f"{EVENT_TYPE}_{quarter}.jsonl.gz"
    )

    if output_path.exists():
        print(f"Файл уже существует: {output_path}")
        return


    with httpx.Client(
        timeout=120.0,
        follow_redirects=True,
    ) as client:
        response = client.get(archive_info["url"])
        response.raise_for_status()

    output_path.write_bytes(response.content)

    print(f"Сохранено: {output_path}")


if __name__ == "__main__":
    main()