class Bcast:
    def __init__(self) -> None:
        self._flag: bool = False

    def get(self) -> bool:
        return self._flag

    def set(self, flag: bool) -> None:
        self._flag = flag

    def toggle(self) -> None:
        self._flag ^= True


bcast_var: Bcast = Bcast()
