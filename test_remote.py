import asyncio
from client import CsvCleanerEnv
try:
    from server.tasks import TASKS
except ImportError:
    TASKS = {"fix_column_types": None, "clean_missing_duplicates": None, "full_pipeline": None}

async def test_space():
    url = "https://printf-sourav-csv-dc-env.hf.space"
    print(f"Connecting to {url}...")
    try:
        env = CsvCleanerEnv(base_url=url)
        for task_name in TASKS.keys():
            print(f"\\nTesting task: {task_name}")
            result = await env.reset(task=task_name)
            print(f"✅ Reset successful. Result: {result}")
            
            # test one simple MCP call
            tools = await env.list_tools()
            if "get_dataset_info" in tools:
                print("✅ Toolkit active.")
                
        await env.close()
        print("\\nAll tasks reachable and responsive in HF Space!")
    except Exception as e:
        print(f"Failed to connect or test: {e}")

if __name__ == "__main__":
    asyncio.run(test_space())
