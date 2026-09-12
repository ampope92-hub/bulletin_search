"""Newsletter-issue corpus: each issue bundles many short, topically distinct items.

An item is generated compositionally from slot vocabularies, so items are unique but
share individual terms with many others -- lexical overlap alone cannot identify one.
Each vocabulary entry is (passage_form, query_form) so the query wording can be pushed
away from the passage wording without changing the referent.
"""
import random

V = {
 "thing": [("silk altar frontal","silk altar frontal"),("linen sampler","linen sampler"),
         ("oak chest","oak chest"),("vellum choir book","vellum choir book"),
         ("bronze votive figure","bronze votive figure"),("porcelain dish","porcelain dish"),
         ("silver caddy spoon","silver caddy spoon"),("ivory hair comb","ivory hair comb"),
         ("lacquer writing box","lacquer writing box"),("plaster wall relief","plaster wall relief"),
         ("leather bookbinding","leather bookbinding"),("glass roundel","glass roundel"),
         ("pewter tankard","pewter tankard"),("papier-mache tea tray","papier-mache tea tray"),
         ("marble portrait bust","marble portrait bust"),("wool tapestry","wool tapestry")],
 "damage": [("flaking paint","a colour layer lifting away from its base"),
            ("cupped and tented ground","preparatory layers curling into ridges"),
            ("powdering pigment","pigment turning to loose dust"),
            ("blistered varnish","bubbles raised under the surface coating"),
            ("cracked gilding","a broken gold leaf finish"),
            ("delaminating veneer","a facing sheet coming away in layers"),
            ("crizzled surface","a fine network of splitting across the surface"),
            ("brittle fibres","threads that had gone dry and snapped easily")],
 "agent": [("a dilute acrylic resin","a weak synthetic polymer solution"),
           ("a warm fish glue","a heated adhesive rendered from fish"),
           ("a seaweed-derived size","a binder made from marine algae"),
           ("a lime nanoparticle dispersion","a suspension of very fine lime in spirit"),
           ("an isinglass solution","a clarified glue prepared from swim bladders"),
           ("a cellulose ether gel","a thickened plant-cellulose preparation"),
           ("a wheat starch paste","a paste cooked from grain starch"),
           ("a microcrystalline wax","a fine-grained synthetic wax")],
 "method": [("through fine tissue","by working it in beneath a sheet of thin paper"),
            ("by nebuliser in several passes","as a fine mist in repeated light coats"),
            ("under a heated spatula","with a warmed hand tool pressed over the area"),
            ("with a fine brush under magnification","using a small brush while viewed closely"),
            ("by capillary feed along the edge","letting it draw in along the border by itself"),
            ("under gentle vacuum suction","while a light suction held the surface down")],
 "soil": [("greasy handling grime","oily marks left by past handling"),
          ("sooty deposits","black residue from smoke"),
          ("a discoloured natural resin coating","a yellowed layer of tree resin"),
          ("hard calcareous crusts","tough chalky encrustation"),
          ("adhesive residue from old tape","glue left behind by removed tape"),
          ("insect frass","waste matter left by insects")],
 "cleaner": [("a rigid chelating gel","a stiff gel holding a metal-binding agent"),
             ("grated vinyl crumb","crumbled plastic granules"),
             ("a buffered enzyme solution","a protein-digesting liquid at controlled alkalinity"),
             ("a pulsed laser at low fluence","a focused light beam at gentle energy"),
             ("a cellulose poultice","a damp absorbent pack"),
             ("swabs of a hydrocarbon and alcohol blend","cotton swabs carrying mixed spirits")],
 "method2": [("in one direction only","always stroking the same way"),
             ("from the least sensitive area outward","starting where the risk was lowest and moving out"),
             ("in short overlapping passes","in brief strokes that each covered the last"),
             ("with frequent renewal of the swab","changing the swab very often")],
 "instr": [("X-ray fluorescence","portable elemental scanning"),
           ("infrared spectroscopy","light-absorption analysis"),
           ("radiography","imaging that sees below the surface"),
           ("electron microscopy of a cross-section","microscopic study of a cut sample"),
           ("polarised light microscopy","microscopy using polarised illumination"),
           ("Raman microscopy","laser scattering microanalysis")],
 "finding": [("a mercury-based red beneath a later overpaint","a red mercury compound hidden under later paint"),
             ("a double ground of chalk under lead white","two preparatory layers, chalk below and lead white above"),
             ("a drying oil bound with a little pine resin","an oil medium carrying a small share of tree resin"),
             ("a reworked limb position in the underdrawing","an arm first sketched in a different place"),
             ("bast fibre in the warp and protein fibre in the weft","plant thread one way and animal thread the other"),
             ("an arsenic-bearing green in the border","a green containing arsenic along the edge")],
 "mount": [("a carved inert foam cradle","a hollowed support of chemically stable foam"),
           ("a sunken window mat","a mount with a recessed opening"),
           ("a coated brass tension strap","a metal strap fitting with a protective finish"),
           ("a barrier-film enclosure with an oxygen scavenger","a sealed gas-tight package holding an oxygen absorber"),
           ("a drop-back clamshell box","a hinged box that opens flat"),
           ("an aluminium honeycomb panel","a lightweight metal backing panel")],
 "padding": [("washed cotton jersey","laundered knitted cotton"),
             ("unbuffered tissue","tissue carrying no alkaline reserve"),
             ("polyester wadding","synthetic fibre batting"),
             ("silk crepeline","a very fine sheer silk")],
 "rh": [("forty-five per cent","a middling level of moisture"),
        ("fifty-five per cent","a slightly damp level of moisture"),
        ("thirty-five per cent","a fairly dry level of moisture"),
        ("sixty per cent","a distinctly humid level of moisture")],
 "lux": [("a fifty lux limit","a very low light ceiling"),
         ("a hundred and fifty lux limit","a moderate light ceiling"),
         ("filtered daylight only","daylight passed through a filter and nothing else"),
         ("a two hundred lux limit with annual rotation","a higher light ceiling with yearly rest periods")],
 "breakage": [("a through crack","a split running right through"),
              ("a detached handle","a handle that had come off"),
              ("an old split along the grain","an earlier lengthwise split"),
              ("a shattered rim","a rim broken into pieces"),
              ("a failed earlier repair","a previous mend that had given way")],
 "joiner": [("hidden hardwood pegs","concealed wooden pins"),
            ("a pin of matching alloy","a metal pin of the same composition"),
            ("thread-by-thread bridging","joining one fibre end at a time"),
            ("a shaped balsa fill","a fitted piece of lightweight wood"),
            ("a reversible animal glue","an adhesive that can later be undone")],
 "coating": [("a sprayed low-molecular-weight resin","a thin sprayed film of light resin"),
             ("a buffed microcrystalline wax","a polished fine synthetic wax"),
             ("a brushed dammar varnish","a brush-applied natural tree varnish"),
             ("a matte cellulose ether film","a non-glossy plant-cellulose film")],
 "pretreat": [("corrosion inhibition with an azole","chemical suppression of corrosion by an azole compound"),
              ("removal of the earlier coating","stripping of the previous finish"),
              ("toning of the fills with dry pigment","colouring the fills using powdered pigment"),
              ("consolidation of the powdering surface","strengthening of the dusty surface")],
 "pest": [("webbing clothes moth","a fabric-eating moth"),
          ("powderpost beetle","a wood-boring beetle"),
          ("silverfish","a paper-grazing bristletail"),
          ("carpet beetle larvae","the grubs of a carpet-eating beetle")],
 "pesttreat": [("a fortnight in a low-oxygen atmosphere","two weeks in an oxygen-depleted enclosure"),
               ("controlled freezing with staged warming","deep chilling followed by gradual return to room temperature"),
               ("isolation and monitoring with blunder traps","quarantine watched by adhesive floor traps"),
               ("localised heat treatment","warming of the affected area only")],
}

# Each object belongs to a material class, and many treatments only make sense on some of
# them: a clothes moth does not infest a glass roundel, and a grain split does not happen in
# porcelain. Slot values are filtered by the class of the object already chosen, so every
# generated item reads as something a conservator could actually have written.
CLASS = {
    "silk altar frontal": "textile", "linen sampler": "textile", "wool tapestry": "textile",
    "oak chest": "wood", "lacquer writing box": "wood",
    "vellum choir book": "book", "leather bookbinding": "book",
    "bronze votive figure": "metal", "silver caddy spoon": "metal", "pewter tankard": "metal",
    "porcelain dish": "ceramic", "glass roundel": "ceramic",
    "plaster wall relief": "stone", "marble portrait bust": "stone",
    "ivory hair comb": "organic", "papier-mache tea tray": "organic",
}
ORGANIC = {"textile", "wood", "book", "organic"}
RESTRICT = {
    "flaking paint": {"wood", "stone", "organic", "metal"},
    "cupped and tented ground": {"wood", "stone"},
    "powdering pigment": {"wood", "stone", "organic"},
    "blistered varnish": {"wood", "organic"},
    "cracked gilding": {"wood", "stone", "organic"},
    "delaminating veneer": {"wood"},
    "crizzled surface": {"ceramic"},
    "brittle fibres": {"textile", "book"},
    "insect frass": ORGANIC,
    "grated vinyl crumb": {"book", "stone", "organic"},
    "a buffered enzyme solution": {"textile", "book"},
    "a cellulose poultice": {"ceramic", "stone"},
    "a pulsed laser at low fluence": {"stone", "metal", "ceramic"},
    "a wheat starch paste": {"textile", "book", "organic"},
    "a seaweed-derived size": {"textile", "book", "organic"},
    "an isinglass solution": {"textile", "book", "organic"},
    "a lime nanoparticle dispersion": {"stone"},
    "a warm fish glue": {"textile", "book", "wood", "organic"},
    "a through crack": {"ceramic", "stone", "wood", "metal", "organic"},
    "a detached handle": {"ceramic", "metal", "wood"},
    "an old split along the grain": {"wood"},
    "a shattered rim": {"ceramic", "metal"},
    "hidden hardwood pegs": {"wood"},
    "a pin of matching alloy": {"metal"},
    "thread-by-thread bridging": {"textile", "book"},
    "a shaped balsa fill": {"wood"},
    "a brushed dammar varnish": {"wood", "stone", "organic"},
    "a buffed microcrystalline wax": {"metal", "wood", "stone"},
    "corrosion inhibition with an azole": {"metal"},
    "webbing clothes moth": {"textile", "wood"},
    "powderpost beetle": {"wood"},
    "silverfish": {"book", "organic"},
    "carpet beetle larvae": {"textile"},
    "a sunken window mat": {"book"},
    "a coated brass tension strap": {"textile"},
    "a barrier-film enclosure with an oxygen scavenger": {"metal", "organic", "book"},
    "a drop-back clamshell box": {"book"},
    "an aluminium honeycomb panel": {"stone", "ceramic"},
    "silk crepeline": {"textile"},
    "polyester wadding": {"textile", "organic"},
}

TEMPLATES = [
 ("consolidation",
  "Treatment of the {damage} on the {thing} used {agent}, applied {method}.",
  "{damage} on the {thing}, set down with {agent} {method}",
  ["damage","thing","agent","method"],
  ["Testing preceded each change of material.",
   "Work stopped once further gain would have been marginal.",
   "The treated area was left under light weight overnight.",
   "Photographs record the surface before and after setting down."]),
 ("cleaning",
  "Cleaning removed {soil} from the {thing} with {cleaner}, working {method2}.",
  "{soil} thinned from the {thing} with {cleaner}, {method2}",
  ["soil","thing","cleaner","method2"],
  ["Clearance was checked under magnification at every stage.",
   "A small window was left untouched for future reference.",
   "Solubility was tested at three separate points first.",
   "Residues were rinsed away with deionised water."]),
 ("analysis",
  "{instr} of the {thing} identified {finding}.",
  "{instr} of the {thing} revealing {finding}",
  ["instr","thing","finding"],
  ["Sample sites are marked on the accompanying diagram.",
   "Results were checked against laboratory reference material.",
   "Sampling was held to the minimum needed.",
   "The finding is consistent with the attributed date."]),
 ("housing",
  "The {thing} was rehoused in {mount}, padded with {padding}.",
  "the {thing} rehoused in {mount}, padded with {padding}",
  ["thing","mount","padding"],
  ["Handling is now possible without touching the surface.",
   "The mount maker was consulted before the design was fixed.",
   "The housing carries a printed handling instruction.",
   "The fit was adjusted twice before it was accepted."]),
 ("environment",
  "The {thing} is displayed at {rh} relative humidity under {lux}.",
  "the {thing} kept at {rh} under {lux}",
  ["thing","rh","lux"],
  ["Loggers in the case are downloaded each quarter.",
   "The specification was agreed with the curator.",
   "Conditions are reviewed whenever the gallery layout changes.",
   "A buffer tray sits beneath the display deck."]),
 ("structural",
  "{breakage} in the {thing} was rejoined with {joiner}.",
  "{breakage} in the {thing} mended with {joiner}",
  ["breakage","thing","joiner"],
  ["Alignment was dry-fitted before any adhesive was introduced.",
   "The join is visible on close inspection by design.",
   "Clamping pressure was kept deliberately light.",
   "The repair can be taken apart again if required."]),
 ("coating",
  "{coating} was applied to the {thing} following {pretreat}.",
  "{coating} applied to the {thing} after {pretreat}",
  ["coating","thing","pretreat"],
  ["Gloss was matched to an adjacent untreated area.",
   "Application took place in a filtered air enclosure.",
   "Two thin coats were preferred to one heavy one.",
   "The finish was left to cure for a week before display."]),
 ("pest",
  "An infestation of {pest} in the {thing} was treated by {pesttreat}.",
  "{pest} in the {thing} dealt with by {pesttreat}",
  ["pest","thing","pesttreat"],
  ["No live activity was seen at the follow-up inspection.",
   "Neighbouring storage was surveyed at the same time.",
   "Debris was vacuumed away before treatment began.",
   "Trap counts have stayed at zero since."]),
]

MASTHEAD = ["Conservation Notes", "Studio Bulletin", "Collections Care Digest"]


def make_items(rng, n_items, paraphrase=True):
    items, seen = [], set()
    tries = 0
    while len(items) < n_items and tries < n_items * 50:
        tries += 1
        topic, ptmpl, qtmpl, slots, support = rng.choice(TEMPLATES)
        ti = rng.randrange(len(V["thing"]))
        cls = CLASS[V["thing"][ti][0]]
        pick = {"thing": ti}
        for sl in slots:
            if sl == "thing":
                continue
            ok = [k for k in range(len(V[sl]))
                  if cls in RESTRICT.get(V[sl][k][0], {cls})]
            if not ok:
                break
            pick[sl] = rng.choice(ok)
        if len(pick) != len(slots):
            continue
        key = (topic,) + tuple(sorted(pick.items()))
        if key in seen:
            continue
        seen.add(key)
        pv = {s: V[s][i][0] for s, i in pick.items()}
        lead = ptmpl.format(**pv)
        lead = lead[0].upper() + lead[1:]
        body = " ".join([lead] + rng.sample(support, 3))
        items.append({"topic": topic, "text": body, "pick": pick,
                      "qtmpl": qtmpl, "slots": slots})
    return items


def render_query(item, pmode, rng):
    """pmode: 'none' | 'full' | int n  (paraphrase n randomly chosen slots)"""
    slots = list(item["pick"].items())
    if pmode == "full":
        para = {s for s, _ in slots}
    elif pmode == "none":
        para = set()
    else:
        k = min(int(pmode), len(slots))
        para = set(rng.sample([s for s, _ in slots], k))
    qv = {s: V[s][i][1 if s in para else 0] for s, i in slots}
    return "Which issue describes " + item["qtmpl"].format(**qv) + "?"


def build(seed=0, n_docs=220, items_per_doc=12, n_items=1400, reps=3, paraphrase=True):
    rng = random.Random(seed)
    items = make_items(rng, n_items, paraphrase)
    pool = []
    for i in range(len(items)):
        pool += [i] * reps
    rng.shuffle(pool)
    need = n_docs * items_per_doc
    while len(pool) < need:
        pool.append(rng.randrange(len(items)))
    pool = pool[:need]

    docs, doc_items = [], []
    for d in range(n_docs):
        idxs = pool[d * items_per_doc:(d + 1) * items_per_doc]
        idxs = list(dict.fromkeys(idxs))
        head = f"{rng.choice(MASTHEAD)}, issue {d + 1}"
        parts = [head]
        for n, ii in enumerate(idxs, 1):
            parts.append(f"## Item {n}\n{items[ii]['text']}")
        docs.append("\n\n".join(parts))
        doc_items.append(idxs)
    where = {}
    for d, idxs in enumerate(doc_items):
        for pos, ii in enumerate(idxs):
            where.setdefault(ii, []).append((d, pos))
    return docs, items, where
