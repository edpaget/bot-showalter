import sys

from fantasy_baseball_manager.marcel.batting import project_batters
from fantasy_baseball_manager.marcel.data_source import PybaseballDataSource
from fantasy_baseball_manager.marcel.pitching import project_pitchers


def main() -> None:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2025

    print(f"MARCEL projections for {year}")
    print(f"Using data from {year - 3}-{year - 1}\n")

    data_source = PybaseballDataSource()

    print("Projecting batters...")
    batting = project_batters(data_source, year)
    batting.sort(key=lambda p: p.hr, reverse=True)

    print(f"\nTop 20 projected HR hitters ({len(batting)} total batters):")
    print(f"{'Name':<25} {'Age':>3} {'PA':>6} {'HR':>5} {'AVG':>6} {'OBP':>6} {'SB':>5}")
    print("-" * 60)
    for p in batting[:20]:
        avg = p.h / p.ab if p.ab > 0 else 0
        obp = (p.h + p.bb + p.hbp) / p.pa if p.pa > 0 else 0
        print(f"{p.name:<25} {p.age:>3} {p.pa:>6.0f} {p.hr:>5.1f} {avg:>6.3f} {obp:>6.3f} {p.sb:>5.1f}")

    print("\nProjecting pitchers...")
    pitching = project_pitchers(data_source, year)
    pitching.sort(key=lambda p: p.so, reverse=True)

    print(f"\nTop 20 projected SO pitchers ({len(pitching)} total pitchers):")
    print(f"{'Name':<25} {'Age':>3} {'IP':>6} {'ERA':>5} {'WHIP':>5} {'SO':>5} {'HR':>4}")
    print("-" * 58)
    for p in pitching[:20]:
        print(f"{p.name:<25} {p.age:>3} {p.ip:>6.1f} {p.era:>5.2f} {p.whip:>5.3f} {p.so:>5.1f} {p.hr:>4.1f}")


if __name__ == "__main__":
    main()
