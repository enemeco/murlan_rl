"""Albanian display names, assigned once per match."""
import secrets

ALBANIAN_NAMES = (
    "Arben", "Arbër", "Dritan", "Ilir", "Luan", "Besnik", "Blerim", "Flamur",
    "Erion", "Endrit", "Bora", "Elira", "Drita", "Blerta", "Arta", "Vjollca",
    "Teuta", "Shpresa", "Liridona", "Besa",
)


def new_player_names(human_seat, previous=()):
    available = [name for name in ALBANIAN_NAMES if name not in previous]
    chosen = iter(secrets.SystemRandom().sample(available, 3))
    return ["Ti" if seat == human_seat else next(chosen) for seat in range(4)]
