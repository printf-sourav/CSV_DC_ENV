import asyncio
from client import CsvCleanerEnv

async def main():
    env = CsvCleanerEnv(base_url='http://localhost:8000')
    await env.connect()
    print('Connected!')
    await env.close()

if __name__ == "__main__":
    asyncio.run(main())
