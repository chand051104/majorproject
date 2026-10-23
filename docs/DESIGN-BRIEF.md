# UrbanGuard dashboard redesign — approval brief

**Research.** I reviewed Awwwards UI design gallery pages [1–5](https://www.awwwards.com/websites/ui-design/) and opened eight listed sites. Notes below describe patterns only; none of their copy, assets, branding, or exact layouts will be reused.

| Site | Pattern worth learning from |
|---|---|
| [Tipalti](https://tipalti.com/) | Product-first hierarchy, compact feature navigation, interface previews and trust evidence beneath the hero. |
| [Two-Do](https://twodo.design/) | Status is written as well as colored; small, consistent labels; reduced-motion treatment; mobile prototype stays readable. |
| [PalaRaiffeisen](https://palaraiffeisen.ch/) | Short top-level navigation, strong place imagery, useful facts surfaced as repeated metrics. |
| [Fitosauna](https://fitosauna.com/) | Clear product hero and CTA, numbered benefit groups, comparison content broken into scannable chunks. |
| [Nova Dental](https://nova-dental-dev.webflow.io/) | Clear section hierarchy, service cards, repeated appointment CTA and trust content. |
| [Garaje Inspiration Fest](https://gif.garajedeideas.com/) | Event-led signup form and a distinctive type treatment; its experimental motion is not appropriate for the dashboard. |
| [Relevo & Vinco](https://relevovinco.com/) | Material texture and paced storytelling are memorable; keep the dashboard’s surfaces restrained and fast. |
| [Cyera AI Guardian](https://secure-the-unknown.cyera.com/) | Guided progression makes a complex topic approachable; its 3D game experience is intentionally excluded from this app. |

**Patterns synthesized:** a clear title and one dominant action; short, predictable navigation; consistent type and spacing; modular cards with restrained borders; status communicated in words and color; motion that reinforces a state; responsive layouts that preserve the primary task.

**Chosen direction — civic operations console.** Keep the map and assessment data dominant. Replace the current bright gradient/indigo treatment with a calm, high-contrast field-report palette and consistent surface system. This is a single-screen dashboard, so the redesign will not invent destinations for the currently inactive rail icons.

**Palette:** ink `#183033`, body `#35494B`, muted `#5D7072`, canvas `#F3F6F5`, surface `#FFFFFF`, border `#D7E1DE`, primary teal `#145D5A`, deeper teal `#104744`, success `#166534`, warning `#92400E`, error/high risk `#B42318`, medium risk `#A35212`. Use dark text on warm accent backgrounds for contrast.

**Type and spacing:** one free system sans stack (`Inter, ui-sans-serif, system-ui, sans-serif`; no paid font or external font request). Scale: 12/14/16/20/24/32 px. Use a 4 px base scale (`4, 8, 12, 16, 24, 32, 48, 64`).

**Component rules:** 8 px control and 12 px card radii; 1 px borders; subtle shadow only for floating map controls; primary teal, outlined secondary, quiet ghost actions; visible 3 px focus ring; inputs and buttons at least 44 px tall; map risk markers retain their existing meaning. Keep existing IDs, routes, API calls and data flow.

**Motion and responsive treatment:** only 160–200 ms opacity/transform transitions for hover, focus and opening an existing detail surface; no scroll effects or 3D. Respect `prefers-reduced-motion`. At 375 px, use a compact two-row header, keep the map usable above the assessment details, then stack cards in one column with no horizontal overflow. At 768 px, give the map and details balanced space. At 1440 px, use a wide map with a stable, scrollable assessment column. The current app has one dashboard route, so no new navigation destinations are proposed.

**Before screenshots:** [375 px](screenshots/before-375.png) · [768 px](screenshots/before-768.png) · [1440 px](screenshots/before-1440.png). The 375 px baseline currently overflows horizontally; the browser recorded no console errors at the three widths.

**Approval:** direction approved by the user before implementation. The changes retain the existing one-screen dashboard and responsive map/details flow.

**After screenshots:** [375 px](screenshots/after-375.png) · [768 px](screenshots/after-768.png) · [1440 px](screenshots/after-1440.png). The redesigned dashboard fits all three widths without horizontal overflow.
