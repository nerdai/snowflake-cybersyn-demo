from asyncio import Lock


class AsyncSafeList:
    def __init__(self):
        self._list = list()
        self._lock = Lock()

    def __aiter__(self):
        return aiter(self._list)

    def __iter__(self):
        return iter(self._list)

    async def append(self, value):
        async with self._lock:
            self._list.append(value)

    async def pop(self):
        async with self._lock:
            return self._list.pop()

    async def delete(self, index):
        async with self._lock:
            del self._list[index]

    async def get(self, index):
        async with self._lock:
            return self._list[index]

    async def length(self):
        async with self._lock:
            return len(self._list)
