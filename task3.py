from dataclasses import dataclass
from typing import List


@dataclass
class GameParams:
    """
    Параметры игры.
    Все величины — скаляры, кроме тех, что помечены как списки/матрицы.
    """
    p: float                  # общая цена продукта
    n: int                    # число фирм
    rho: float                
    T: int                    # число периодов (0..T)
    delta: float              # скорость "забывания" бренда
    eta: float                # линейная часть издержек рекламы
    eta_i: List[float]        # максимальные уровни рекламы для каждой фирмы
    eps: float                # коэффициент квадратичных издержек рекламы (ε)
    alpha: float              # эффект собственной рекламы
    beta: float               # сетевой эффект рекламы
    gamma: float              # конкурентный переток внимания
    c0: List[float]           # начальная узнаваемость бренда c_i(0)
    pi_ij: List[List[float]]  # матрица издержек сотрудничества π_ij
    A: List[List[int]]        # матрица смежности сети A_ij (0 или 1)


@dataclass
class GameResult:
    """
    Результат моделирования игры.
    """
    u: List[List[float]]          # u[t][i] — реклама фирмы i в момент t
    y: List[List[float]]          # y[t][i] — прибыль фирмы i в момент t
    total_profit: List[float]     # суммарная дисконтированная прибыль по фирмам


def equilibrium_ad_level(params: GameParams) -> List[float]:
    """
    Считает равновесный уровень рекламы для каждой фирмы.
    Формула:
        u* = (p * alpha - eta) / (2 * eps)
    Затем обрезает в диапазон [0, eta_i] для каждой фирмы.
    """
    base = (params.p * params.alpha - params.eta) / (2.0 * params.eps)

    u_star = []
    for i in range(params.n):
        # Ограничения по мощности рекламы: 0 <= u_i <= eta_i
        u_i = max(0.0, min(params.eta_i[i], base))
        u_star.append(u_i)

    return u_star


def simulate_game(params: GameParams) -> GameResult:
    """
    Моделирование динамики игры при равновесных стратегиях рекламы u_i*.

    Шаги:
    1) Вычисляет u_i* и фиксируем их для всех t.
    2) Запускает цикл по времени и считает:
       - новую узнаваемость c_i(t),
       - прибыль y_i(t),
       - дисконтирует и аккумулирует суммарную прибыль.
    """
    n = params.n
    T = params.T

    # Равновесные уровни рекламы (стратегии фирм)
    u_star = equilibrium_ad_level(params)

    # Инициализация массивов: T+1 строк (t=0..T), n колонок (i=0..n-1)
    u = [[0.0 for _ in range(n)] for _ in range(T + 1)]
    y = [[0.0 for _ in range(n)] for _ in range(T + 1)]
    c = [[0.0 for _ in range(n)] for _ in range(T + 1)]  # узнаваемость брендов

    # Начальная узнаваемость
    for i in range(n):
        c[0][i] = params.c0[i]

    # Суммарная дисконтированная прибыль
    total_profit = [0.0 for _ in range(n)]

    # Основной цикл по времени
    for t in range(T + 1):
        # 1) Фиксируем рекламу в момент t: u_i(t) = u_i*
        for i in range(n):
            u[t][i] = u_star[i]

        # 2) Считаем прибыль y_i(t) для текущего состояния c_i(t) и рекламы u_i(t)
        for i in range(n):
            # Выручка: p * c_i(t)
            revenue = params.p * c[t][i]

            # Стоимость рекламы: eps * u_i^2 + eta * u_i
            adv_cost = params.eps * (u[t][i] ** 2) + params.eta * u[t][i]

            # Издержки сотрудничества: сумма по соседям A_ij * pi_ij
            coop_cost = 0.0
            for j in range(n):
                if params.A[i][j] == 1:
                    coop_cost += params.pi_ij[i][j]

            y[t][i] = revenue - adv_cost - coop_cost

        # 3) Добавляем дисконтированную прибыль к суммарной
        discount = params.rho ** t
        for i in range(n):
            total_profit[i] += discount * y[t][i]

        # 4) Переход к следующему периоду: считаем c_i(t+1)
        if t < T:
            for i in range(n):
                # Собственная часть: "память" бренда
                next_c = (1.0 - params.delta) * c[t][i]

                # Эффект собственной рекламы
                next_c += params.alpha * u[t][i]

                # Сетевой эффект рекламы и конкуренции
                sum_beta = 0.0
                sum_gamma = 0.0
                for j in range(n):
                    if i == j:
                        continue
                    if params.A[i][j] == 1:
                        # Влияние рекламы соседей (совместные кампании)
                        sum_beta += params.beta * u[t][j]
                        # Конкурентный переток внимания (выравнивание узнаваемости)
                        sum_gamma += params.gamma * (c[t][j] - c[t][i])

                next_c += sum_beta + sum_gamma

                # Не даём узнаваемости уйти в минус
                c[t + 1][i] = max(0.0, next_c)

    return GameResult(u=u, y=y, total_profit=total_profit)


def print_results(params: GameParams, result: GameResult) -> None:
    """
    Красивый вывод результатов в виде таблиц:
    для каждой фирмы i печатаем:
      t | u_i(t) | y_i(t)
    и суммарную прибыль.
    """
    n = params.n
    T = params.T

    for i in range(n):
        print(f"\n=== Фирма {i + 1} ===")
        print("t\tu_i(t)\ty_i(t)")
        for t in range(T + 1):
            u_it = result.u[t][i]
            y_it = result.y[t][i]
            print(f"{t}\t{u_it:.4f}\t{y_it:.4f}")
        print(f"Суммарная дисконтированная прибыль: {result.total_profit[i]:.4f}")


# ------------------------ Пример запуска ------------------------ #
if __name__ == "__main__":
    # Пример параметров 
    T = 5
    n = 3

    # Полная сеть: каждая фирма связана со всеми, кроме себя
    A = [[1 if i != j else 0 for j in range(n)] for i in range(n)]

    # Издержки сотрудничества: по 0.5 за каждый активный канал
    pi_ij = [[0.5 if A[i][j] == 1 else 0.0 for j in range(n)] for i in range(n)]

    params = GameParams(
        p=10.0,
        n=n,
        rho=0.9,
        T=T,
        delta=0.2,
        eta=1.0,
        eta_i=[2.0, 1.5, 1.0],
        eps=0.3,
        alpha=0.5,
        beta=0.2,
        gamma=0.1,
        c0=[1.0, 0.5, 0.2],
        pi_ij=pi_ij,
        A=A,
    )

    result = simulate_game(params)
    print_results(params, result)
