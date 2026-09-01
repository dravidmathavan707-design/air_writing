from hero.heroes import cosmic_hero, energy_hero, iron_hero, ninja_hero, tech_hero, web_hero

STYLES = {
    iron_hero.NAME: iron_hero,
    web_hero.NAME: web_hero,
    tech_hero.NAME: tech_hero,
    energy_hero.NAME: energy_hero,
    cosmic_hero.NAME: cosmic_hero,
    ninja_hero.NAME: ninja_hero,
}

HERO_ORDER = ("iron", "web", "tech", "energy", "cosmic", "ninja")
