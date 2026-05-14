## Aesthetic Vision for Wikipedia Golf

### Design Philosophy
Wikipedia Golf embodies **Augusta National elegance**—patrician, timeless, and quietly confident. The aesthetic whispers prestige rather than shouting it. Every element should feel like a perfectly manicured fairway: immaculate, uncluttered, and purposeful.

### Color system
Typography and chrome should **read green-first**, without a parallel gray vocabulary (mist, slate, charcoal boxes).

- **Paper & surface** — Warm ivory backdrop (`#f8f8f6`-class) plus pure white for elevated panels and inputs keeps the baseline calm.
- **Ink** — Body text favors a **deep green-black** tint so prose feels club-stationery, not neutral dashboard copy.
- **Muted text** — Use **Augusta green at fractional opacity** on light surfaces instead of unrelated cool grays.
- **Rails & sheets** — Hairlines between items and rest-state input edges use **thin green-transparent lines**. Soft shadows borrow the same hue at low intensity so elevations never read as smoky gray.
- **Interaction** — **Augusta Green** anchors primary fills; **Billiard Green** supports hover/active depth in the same family.
- **Championship yellow / gold** — Reserved for deliberate ceremony when we introduce it—not required for baseline UI chrome.
- **Errors** — A restrained red set remains for readability and urgency.

Implementation detail lives as CSS tokens in `assets/custom.css` (`--wg-*`), not duplicated here numerically—palette iteration should happen once in code.

### Animation & Transitions
- Subtle transitions (0.15s–0.2s ease) on:
  - Border colors (input focus)
  - Background colors (hover states)
  - Opacity (optional, for result appearance)
- No jarring movements or bouncy animations
- Smooth, professional, restrained

### Iconography
- Minimal icons. Prefer text or simple geometric shapes.
- Arrow (→) as primary visual connector
