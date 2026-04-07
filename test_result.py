import asyncio
from client import CsvCleanerEnv

async def main():
    env = CsvCleanerEnv(base_url='http://localhost:8000')
    await env.connect()
    # Need to await reset
    obs = await env.reset(task_name="fix_column_types")
    print(dir(obs))
    print(obs)
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
