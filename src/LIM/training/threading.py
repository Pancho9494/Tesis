from concurrent.futures import ThreadPoolExecutor

backup_executor = ThreadPoolExecutor(max_workers=2)
