# import time
# from concurrent.futures import ProcessPoolExecutor
# import concurrent
# from multiprocessing import freeze_support
#
#
# def my_parallel_process(i: int):
#     time.sleep(2)
#     print(f"josh was here ({i})")
#
#
# def run_processes():
#     with ProcessPoolExecutor(max_workers=3) as executor:
#         for i in range(0, 5):
#             executor.submit(my_parallel_process(i))
#
#
#
#
# if __name__ == '__main__':
#     # freeze_support()
#     run_processes()


import time
import concurrent.futures
import urllib.request

URLS = ['http://www.foxnews.com/',
        'http://www.cnn.com/',
        'http://europe.wsj.com/',
        'http://www.bbc.co.uk/',
        'http://nonexistant-subdomain.python.org/']

# Retrieve a single page and report the URL and contents
def load_url(url, timeout):
    time.sleep(5)
    with urllib.request.urlopen(url, timeout=timeout) as conn:
        return conn.read()

# We can use a with statement to ensure threads are cleaned up promptly
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    # Start the load operations and mark each future with its URL
    future_to_url = {}
    for url in URLS:
        future = executor.submit(load_url, url, 60)
        future_to_url[future] = url
    # future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
    print("waiting")
    for future in concurrent.futures.as_completed(future_to_url):
        url = future_to_url[future]
        try:
            data = future.result()
        except Exception as exc:
            print('%r generated an exception: %s' % (url, exc))
        else:
            print('%r page is %d bytes' % (url, len(data)))
