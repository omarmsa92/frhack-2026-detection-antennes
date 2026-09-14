"""
Teste la logique pure d'app.py en extrayant les fonctions reelles du fichier
source (via AST), sans avoir besoin de streamlit/plotly.
"""
import ast, sys, math, textwrap
import numpy as np
sys.path.insert(0, '/home/claude/anfr')
from geoloc import pixels_to_gps, lambert93_convergence, ground_extent_from_mercator

SRC = open('/home/claude/anfr/app.py').read()
tree = ast.parse(SRC)
WANT = {'click_grid', 'parse_selection', 'geometry_for_site'}
ns = {'np': np, 'lambert93_convergence': lambert93_convergence,
      'ground_extent_from_mercator': ground_extent_from_mercator}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        node.decorator_list = []          # retire @st.cache_data
        exec(compile(ast.Module([node], []), '<app>', 'exec'), ns)
click_grid, parse_selection, geometry_for_site = (ns[n] for n in
    ('click_grid', 'parse_selection', 'geometry_for_site'))
print(f"fonctions extraites d'app.py : {sorted(WANT)}\n")

# --------------------------------------------------------------------------
print("="*74); print("TEST 1 - la grille couvre bien toute l'image"); print("="*74)
for step in (2, 4, 8, 16):
    gx, gy = click_grid(1024, 1024, step)
    n = len(gx)
    # pire distance entre un pixel quelconque et le point de grille le plus proche
    worst = math.hypot(step/2, step/2)
    assert gx.min() == step/2 and gx.max() <= 1024
    print(f"  pas {step:2d} px : {n:>6} points | erreur max {worst:5.2f} px "
          f"= {worst*0.2:4.2f} m @20cm | payload ~{n*2*8/1e6:.1f} Mo")
gx, gy = click_grid(1024, 1024, 4)
assert len(gx) == 256*256

# --------------------------------------------------------------------------
print("\n"+"="*74); print("TEST 2 - parsing des evenements de selection"); print("="*74)
cases = [
    ("rien",                None,                                        (None,None,None)),
    ("selection vide",      {"selection": {"points": [], "box": []}},    (None,None,None)),
    ("clic sur la grille",  {"selection": {"points": [
                              {"curve_number": 1, "x": 612.0, "y": 388.0}]}}, (612.0,388.0,None)),
    ("clic sur la croix",   {"selection": {"points": [
                              {"curve_number": 2, "x": 10.0, "y": 10.0}]}}, (None,None,None)),
    ("rectangle",           {"selection": {"box": [
                              {"x": [600.0, 640.0], "y": [400.0, 360.0]}]}}, (620.0,380.0,(600.0,360.0,640.0,400.0))),
    ("rectangle inverse",   {"selection": {"box": [
                              {"x": [640.0, 600.0], "y": [360.0, 400.0]}]}}, (620.0,380.0,(600.0,360.0,640.0,400.0))),
]
for name, ev, expected in cases:
    got = parse_selection(ev)
    ok = "OK " if got == expected else "FAIL"
    print(f"  [{ok}] {name:22s} -> {got}")
    assert got == expected, (name, got, expected)

class EventObj:                      # Streamlit renvoie un objet, pas un dict
    selection = {"points": [{"curve_number": 1, "x": 100.0, "y": 200.0}]}
    def get(self, k, d=None): raise AssertionError("ne doit pas utiliser .get")
got = parse_selection(EventObj())
print(f"  [OK ] acces par attribut     -> {got}")
assert got == (100.0, 200.0, None)

# --------------------------------------------------------------------------
print("\n"+"="*74); print("TEST 3 - geometrie selon le CRS"); print("="*74)
lat, lon, W = 49.95, 1.658814, 1024

ns['pixel_size_manual'], ns['use_convergence'], ns['bbox_span'] = None, False, 204.8
e, g, c = geometry_for_site(lat, lon, W)
print(f"  EPSG:3857  emprise {e:7.2f} m | {g*100:5.2f} cm/px | convergence {c:+.2f} deg")
assert abs(e - 204.8*math.cos(math.radians(lat))) < 1e-9 and c == 0.0

ns['pixel_size_manual'], ns['use_convergence'], ns['bbox_span'] = None, True, 204.8
e, g, c = geometry_for_site(lat, lon, W)
print(f"  EPSG:2154  emprise {e:7.2f} m | {g*100:5.2f} cm/px | convergence {c:+.2f} deg")
assert e == 204.8 and abs(c - lambert93_convergence(lon)) < 1e-12

ns['pixel_size_manual'], ns['use_convergence'], ns['bbox_span'] = 0.20, False, None
e, g, c = geometry_for_site(lat, lon, W)
print(f"  manuel     emprise {e:7.2f} m | {g*100:5.2f} cm/px | convergence {c:+.2f} deg")
assert e == 204.8 and g == 0.20

# --------------------------------------------------------------------------
print("\n"+"="*74); print("TEST 4 - chaine complete clic -> GPS"); print("="*74)
ns['pixel_size_manual'], ns['use_convergence'], ns['bbox_span'] = None, True, 204.8
e, g, c = geometry_for_site(lat, lon, W)
for px, py in [(512,512), (612,388), (0,0), (1023,1023)]:
    la, lo = pixels_to_gps(px, py, lat, lon, image_size_px=W, extent_m=e,
                           grid_convergence_deg=c)
    d = math.hypot((px-512)*g, (512-py)*g)
    print(f"  px({px:4d},{py:4d}) -> {la:.7f}, {lo:.7f}  (decalage {d:6.1f} m)")
la, lo = pixels_to_gps(512, 512, lat, lon, image_size_px=W, extent_m=e, grid_convergence_deg=c)
assert abs(la-lat) < 1e-12 and abs(lo-lon) < 1e-12, "le centre doit redonner l'ANFR"
print("  centre -> coordonnee ANFR exacte : OK")

# --------------------------------------------------------------------------
print("\n"+"="*74); print("TEST 5 - garde anti-boucle sur la selection"); print("="*74)
state = {"last_sel": None, "pixel": None}
def handle(ev):
    """Reproduit la logique de garde d'app.py."""
    nx, ny, nbox = parse_selection(ev)
    if nx is None: return False
    key = (round(nx,2), round(ny,2), nbox)
    if state["last_sel"] != key:
        state["last_sel"] = key
        state["pixel"] = (round(nx,2), round(ny,2))
        return True          # -> st.rerun()
    return False
ev = {"selection": {"points": [{"curve_number": 1, "x": 612.0, "y": 388.0}]}}
print(f"  1er rerun (nouveau clic)      -> rerun={handle(ev)}  pixel={state['pixel']}")
print(f"  2e rerun (meme selection)     -> rerun={handle(ev)}  (pas de boucle)")
state["pixel"] = (615.0, 390.0)                      # reglage fin par l'utilisateur
print(f"  reglage fin -> pixel={state['pixel']}")
print(f"  3e rerun (selection inchangee)-> rerun={handle(ev)}  pixel={state['pixel']}")
assert state["pixel"] == (615.0, 390.0), "le reglage fin ne doit PAS etre ecrase"
print("  le reglage fin survit aux reruns : OK")

print("\nTOUS LES TESTS PASSENT")
