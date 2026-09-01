from animation.animation_manager import CHARGING, IDLE, READY, RELEASING, AnimationManager
from animation.hand_pose import AnimationPose
from animation.particle_system import Particle, ParticleSystem


def test_particles_die_after_lifetime():
    system = ParticleSystem()
    system.emit(Particle(x=0, y=0, life=0.2, max_life=0.2))
    system.update(0.25)
    assert len(system) == 0


def test_charge_fist_then_open_palm_releases():
    manager = AnimationManager(charge_seconds=0.2, cooldown_seconds=0.2)
    fist = AnimationPose(
        present=True,
        index_tip=(200.0, 220.0),
        palm=(200.0, 280.0),
        wrist=(200.0, 340.0),
        is_fist=True,
    )
    for _ in range(6):
        manager._update_state(0.05, fist)
    assert manager.state in (CHARGING, READY)
    assert manager.charge >= 0.9
    opened = AnimationPose(
        present=True,
        index_tip=(200.0, 220.0),
        palm=(200.0, 280.0),
        wrist=(200.0, 340.0),
        is_open=True,
    )
    manager._update_state(0.05, opened)
    assert manager.state == RELEASING
    assert manager.blast.active


def test_chidori_release_fires_bolts():
    manager = AnimationManager(charge_seconds=0.2, cooldown_seconds=0.2)
    manager.set_effect("chidori")
    fist = AnimationPose(
        present=True,
        index_tip=(200.0, 220.0),
        palm=(200.0, 280.0),
        wrist=(200.0, 340.0),
        is_fist=True,
        angle=0.0,
    )
    for _ in range(6):
        manager._update_state(0.05, fist)
    opened = AnimationPose(
        present=True,
        index_tip=(200.0, 220.0),
        palm=(200.0, 280.0),
        wrist=(200.0, 340.0),
        is_open=True,
        angle=0.0,
    )
    manager._update_state(0.05, opened)
    assert manager.state == RELEASING
    assert manager.chidori.firing
    assert any(b.kind == "spear" for b in manager.chidori.bolts)
    assert any(b.kind == "senbon" for b in manager.chidori.bolts)
    assert any(b.kind == "stream" for b in manager.chidori.bolts)


def test_idle_until_fist():
    manager = AnimationManager()
    idle = AnimationPose(present=True, is_pointing=True, index_tip=(10, 10), palm=(10, 40), wrist=(10, 70))
    manager._update_state(0.1, idle)
    assert manager.state == IDLE
    assert manager.charge == 0.0
