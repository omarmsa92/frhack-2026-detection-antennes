"""Validation de geoloc.py contre une projection Lambert-93 independante."""
import math, sys
import numpy as np
sys.path.insert(0, '/home/claude/anfr')
from geoloc import pixels_to_gps, gps_to_pixels, lambert93_convergence, ground_extent_from_mercator

# ---- projection Lambert-93 de reference, implementee independamment ---------
A=6378137.0; F=1/298.257222101; E2=2*F-F*F; E=math.sqrt(E2)
LAT1,LAT2,LAT0,LON0,X0,Y0=map(math.radians,(44,49,46.5,3,0,0)); X0,Y0=700000.0,6600000.0
def _m(p): return math.cos(p)/math.sqrt(1-E2*math.sin(p)**2)
def _t(p): return math.tan(math.pi/4-p/2)/((1-E*math.sin(p))/(1+E*math.sin(p)))**(E/2)
N_=(math.log(_m(LAT1))-math.log(_m(LAT2)))/(math.log(_t(LAT1))-math.log(_t(LAT2)))
FF=_m(LAT1)/(N_*_t(LAT1)**N_); RHO0=A*FF*_t(LAT0)**N_
def to_l93(lon,lat):
    p=math.radians(lat); rho=A*FF*_t(p)**N_; th=N_*(math.radians(lon)-LON0)
    return X0+rho*math.sin(th), Y0+RHO0-rho*math.cos(th)
def k_l93(lat):
    p=math.radians(lat); return N_*FF*_t(p)**N_/_m(p)

SITES=[(50.117991,1.658814),(49.892376,2.009789),(49.986529,3.193847),
       (50.344028,1.551625),(49.589841,2.666604),(49.950000,2.269600)]
CLICKS=[(512,512),(0,0),(1023,1023),(1023,0),(0,1023),(700,300),(190,880)]

print("="*78); print("TEST 1 - aller-retour pixel -> GPS -> pixel"); print("="*78)
worst=0.0
for lat0,lon0 in SITES:
    for x,y in CLICKS:
        la,lo = pixels_to_gps(x,y,lat0,lon0)
        xr,yr = gps_to_pixels(la,lo,lat0,lon0)
        worst=max(worst, math.hypot(xr-x,yr-y))
print(f"  erreur max aller-retour : {worst:.3e} px  ({worst*0.2*1000:.3e} mm au sol)")
assert worst < 1e-6

print("\n"+"="*78); print("TEST 2 - distance restituee, controlee en Lambert-93"); print("="*78)
print(f"  {'site':>22} {'clic':>12} {'attendu':>10} {'mesure L93':>12} {'ecart':>9}")
worst_d=0.0
for lat0,lon0 in SITES[:3]:
    for x,y in [(1023,0),(700,300),(0,1023)]:
        la,lo = pixels_to_gps(x,y,lat0,lon0)
        e0,n0 = to_l93(lon0,lat0); e1,n1 = to_l93(lo,la)
        k = k_l93((lat0+la)/2)
        d_grid = math.hypot(e1-e0,n1-n0)/k          # grille -> distance au sol
        d_att  = math.hypot((x-512)*0.2,(512-y)*0.2)
        worst_d=max(worst_d,abs(d_grid-d_att))
        print(f"  ({lat0:.4f},{lon0:.4f}) {str((x,y)):>12} {d_att:9.3f}m {d_grid:11.3f}m {(d_grid-d_att)*100:8.2f}cm")
print(f"\n  ecart max : {worst_d*100:.2f} cm sur {d_att:.0f} m  -> {worst_d/d_att*1e6:.1f} ppm")
assert worst_d < 0.02

print("\n"+"="*78); print("TEST 3 - cout des approximations naives"); print("="*78)
lat0,lon0 = 49.95, 1.658814
x,y = 1023, 0        # coin nord-est, le cas le plus defavorable
la,lo = pixels_to_gps(x,y,lat0,lon0)

# (a) terre spherique R=6371km au lieu de l'ellipsoide
R=6371000.0; gsd=0.2
dlat_s=(512-y)*gsd/R; dlon_s=(x-512)*gsd/(R*math.cos(math.radians(lat0)))
la_s,lo_s=lat0+math.degrees(dlat_s), lon0+math.degrees(dlon_s)
e,nn=to_l93(lo,la); es,ns=to_l93(lo_s,la_s)
print(f"  (a) sphere R=6371 km            -> {math.hypot(es-e,ns-nn)*100:7.1f} cm")

# (b) 111320 m/deg, la constante qu'on voit partout
la_c = lat0 + (512-y)*gsd/111320.0
lo_c = lon0 + (x-512)*gsd/(111320.0*math.cos(math.radians(lat0)))
ec,nc=to_l93(lo_c,la_c)
print(f"  (b) constante 111320 m/deg      -> {math.hypot(ec-e,nc-nn)*100:7.1f} cm")

# (c) convergence des meridiens ignoree sur un crop EPSG:2154
for lon_t in (1.40, 2.27, 3.19):
    g = lambert93_convergence(lon_t)
    la_g,lo_g = pixels_to_gps(x,y,lat0,lon_t,grid_convergence_deg=g)
    la_n,lo_n = pixels_to_gps(x,y,lat0,lon_t)
    eg,ng=to_l93(lo_g,la_g); en,nnn=to_l93(lo_n,la_n)
    print(f"  (c) convergence ignoree, lon={lon_t:.2f} (gamma={g:+.2f} deg) -> {math.hypot(eg-en,ng-nnn)*100:7.1f} cm")

# (d) Web Mercator pris pour des metres
vraie = ground_extent_from_mercator(204.8, lat0)
la_w,lo_w = pixels_to_gps(x,y,lat0,lon0,extent_m=vraie)
ew,nw=to_l93(lo_w,la_w)
print(f"  (d) bbox 3857 prise pour 204.8 m -> {math.hypot(ew-e,nw-nn):7.1f} m   <<< ordre de grandeur superieur")
print(f"      (emprise reelle = {vraie:.1f} m, gsd = {vraie/1024*100:.2f} cm/px)")

print("\n"+"="*78); print("TEST 4 - vectorisation et coherence avec le pipeline"); print("="*78)
xs=np.array([0,512,1023,700]); ys=np.array([0,512,1023,300])
la_v,lo_v = pixels_to_gps(xs,ys,49.95,2.27)
print(f"  entree array{xs.shape} -> sortie {type(la_v).__name__}{la_v.shape}  OK")
# le centre doit redonner exactement la coordonnee ANFR
la_c2,lo_c2 = pixels_to_gps(512.0,512.0,49.892376,2.009789)
print(f"  centre (512,512) -> ({la_c2:.9f}, {lo_c2:.9f})  ecart = "
      f"{abs(la_c2-49.892376):.1e}, {abs(lo_c2-2.009789):.1e} deg")
assert abs(la_c2-49.892376)<1e-12 and abs(lo_c2-2.009789)<1e-12

# une boite YOLO du pipeline -> coin superieur gauche en GPS
xc,yc,w,h = 0.490416,0.473668,0.116611,0.150107
x1,y1=(xc-w/2)*1024,(yc-h/2)*1024
print(f"  boite YOLO ANFR_105438 coin haut-gauche px({x1:.1f},{y1:.1f}) -> "
      f"{pixels_to_gps(x1,y1,49.892376,2.009789)}")
print("\nTOUS LES TESTS PASSENT")
