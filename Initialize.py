import os
import glob
import time
import shutil
import subprocess
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import wget
import arrow
import schedule
import numpy as np
import xarray as xr
import meshio
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import ListedColormap, BoundaryNorm

# Global date variable (evaluated dynamically when run_sequence is triggered)
TODAYSDATE = arrow.now().format('YYYYMMDD')

# Environment configuration for emails (Set these in your environment)
SENDER_EMAIL = os.environ.get("MEWFS_SENDER_EMAIL", "")
RECEIVER_EMAIL = os.environ.get("MEWFS_RECEIVER_EMAIL", "")
SMTP_APP_PASSWORD = os.environ.get("MEWFS_SMTP_PASSWORD", "YOUR_GMAIL_APP_PASSWORD")


SENDER_EMAIL = ""
RECEIVER_EMAIL = ""
# Uses environment variable, falling back to working app password
SMTP_APP_PASSWORD = os.environ.get("MEWFS_SMTP_PASSWORD", "")


def send_email(subject: str, body: str, retries: int = 3):
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    for attempt in range(1, retries + 1):
        try:
            server = smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30)
            server.login(SENDER_EMAIL, SMTP_APP_PASSWORD)
            server.send_message(msg)
            server.quit()
            print(f"Email '{subject}' sent successfully.")
            return
        except Exception as e:
            print(f"Attempt {attempt}/{retries} failed to send email '{subject}': {e}")
            if attempt < retries:
                time.sleep(5)


def send_startup_email():
    send_email("MEWFM BEGINS", "MEFWFM FOR TODAY BEGINS SUCCESSFULLY")


def send_conclude_email():
    send_email("MEWFM CONCLUDES", "MEFWFM FOR TODAY CONCLUDED SUCCESSFULLY")


def get_gfs_grib():
    print("Attempting to execute task downloading global data at:", time.strftime("%H:%M:%S"))
    os.makedirs(TODAYSDATE, exist_ok=True)

    for i in range(0, 24, 1):
        print(f'DOWNLOADING FORECAST : {i:03d}z')
        url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/gfs/prod/gfs.{TODAYSDATE}/00/wave/gridded/gfswave.t00z.global.0p16.f{i:03d}.grib2"
        print(f'Downloading forecast URL: {url}')
        try:
            wget.download(url, out=TODAYSDATE)
            print('\n')
        except Exception as e:
            print(f"\nFailed to download {url}: {e}")


def get_bund_valss():
    grib_files = glob.glob(os.path.join(TODAYSDATE, '*.grib2'))
    bunds = [101, 102, 103, 104]

    for filepath in grib_files:
        print(f'LOADING GRIB FILE {filepath}')
        ds = xr.open_dataset(filepath)

        bund_forcing_swh = []
        bund_forcing_perpw = []
        bund_forcing_dirpw = []

        for k in bunds:
            bund_file = f'INITIALISATION/FORCING_BUND_COORDS/{k}.csv'
            current_bund = np.loadtxt(bund_file, delimiter=',', skiprows=1)

            bund_all_swh = []
            bund_all_perpw = []
            bund_all_dirpw = []

            for j in current_bund:
                swh = ds.sel(latitude=j[3], longitude=j[2], method="nearest")['swh'].values
                perpw = ds.sel(latitude=j[3], longitude=j[2], method="nearest")['perpw'].values
                dirpw = ds.sel(latitude=j[3], longitude=j[2], method="nearest")['dirpw'].values

                bund_all_swh.append(swh)
                bund_all_perpw.append(perpw)
                bund_all_dirpw.append(dirpw)

            bund_forcing_swh.append(np.max(bund_all_swh))
            bund_forcing_perpw.append(np.max(bund_all_perpw))
            bund_forcing_dirpw.append(np.max(bund_all_dirpw))

        ds.close()

        np.savetxt(f'{filepath}_FORCING_swh.csv', np.array(bund_forcing_swh), delimiter=',')
        np.savetxt(f'{filepath}_FORCING_perpw.csv', np.array(bund_forcing_perpw), delimiter=',')
        np.savetxt(f'{filepath}_FORCING_dirpw.csv', np.array(bund_forcing_dirpw), delimiter=',')


def get_winds():
    print("EVALUATING FOR WIND")
    grib_files = glob.glob(os.path.join(TODAYSDATE, '*.grib2'))

    for filepath in grib_files:
        print(f"EVALUATING FOR WIND AT GRIB FILE {filepath}")
        ds = xr.open_dataset(filepath)

        wind_val = ds.sel(latitude=1.90, longitude=73.40, method="nearest")['ws'].values
        wind_dir = ds.sel(latitude=1.90, longitude=73.40, method="nearest")['wdir'].values
        ds.close()

        with open(f'{filepath}_FORCING_wind.csv', "w") as f:
            f.write(f"{wind_val}\n")
        with open(f'{filepath}_FORCING_dir.csv', "w") as f:
            f.write(f"{wind_dir}\n")


def write_run_file():
    all_times = [f'f{i:03d}' for i in range(0, 24, 1)]

    for j in all_times:
        dir_file = os.path.join(TODAYSDATE, f'gfswave.t00z.global.0p16.{j}.grib2_FORCING_dirpw.csv')
        period_file = os.path.join(TODAYSDATE, f'gfswave.t00z.global.0p16.{j}.grib2_FORCING_perpw.csv')
        swh_file = os.path.join(TODAYSDATE, f'gfswave.t00z.global.0p16.{j}.grib2_FORCING_swh.csv')
        wind_file = os.path.join(TODAYSDATE, f'gfswave.t00z.global.0p16.{j}.grib2_FORCING_wind.csv')
        wind_dir_file = os.path.join(TODAYSDATE, f'gfswave.t00z.global.0p16.{j}.grib2_FORCING_dir.csv')

        direction = np.loadtxt(dir_file, delimiter=',')
        period = np.loadtxt(period_file, delimiter=',')
        wave_height = np.loadtxt(swh_file, delimiter=',')
        wind_speed = np.loadtxt(wind_file, delimiter=',')
        wind_direction = np.loadtxt(wind_dir_file, delimiter=',')

        print(f"Writing run files for forecast hour {j}")
        swn_path = os.path.join(TODAYSDATE, f'{j}.swn')

        with open(swn_path, "w") as f:
            f.write(f'$****************LAAMU {TODAYSDATE} {j} ****************************************\n\n')
            f.write(f"PROJ 'LAAMU_{TODAYSDATE}' '{j}'\n\n")
            f.write('SET depmin 0.0 NAUTICAL\n\n')
            f.write('CGRID UNSTRUCTURED CIRCLE 36 0.0521 1. 31\n\n')
            f.write("READ UNSTRUCTURED triangle '../INITIALISATION/MESH_BATHY/TEST.4'\n\n")
            f.write('INPgrid BOTtom REGular 295668.0 191709.0 0. 999 999 55. 55. EXCeption 999999.00\n')
            f.write("READinp BOTtom -1. '../INITIALISATION/MESH_BATHY/clippedbathy_1000_1000.csv' 4 FREE\n\n")
            f.write('BOUN SHAPE JON 3.3 PEAK DSPR DEGREES\n')
            f.write(f'BOUN SIDE 101 CONST PAR {wave_height[0]} {period[0]} {direction[0]} 30\n')
            f.write(f'BOUN SIDE 102 CONST PAR {wave_height[1]} {period[1]} {direction[1]} 30\n')
            f.write(f'BOUN SIDE 103 CONST PAR {wave_height[2]} {period[2]} {direction[2]} 30\n')
            f.write(f'BOUN SIDE 104 CONST PAR {wave_height[3]} {period[3]} {direction[3]} 30\n\n')
            f.write(f'WIND {wind_speed} {wind_direction}\n\n')
            f.write('GEN3 JANSSEN 4.5 0.5 AGROW 0.003\nQUAD\nBREA\nWCAP\n\n')
            f.write('$********************* OUTPUT REQUESTS *************************\n\n')
            f.write(f"BLOCK 'COMPGRID' NOHEAD 'Laamu_{TODAYSDATE}_{j}.vtu' LAYOUT 3 XP YP BOTLEV WATLEV DEPTH HS TM01 DSPR PDIR VEL FRCOEFF TPS URMS\n")
            f.write('TEST 1,0\n\nCOMPUTE\nSTOP')


def run_files():
    original_dir = os.getcwd()
    all_swan_files = [f'f{i:03d}.swn' for i in range(0, 24, 1)]

    run_script = os.path.join(TODAYSDATE, 'run_all.sh')
    with open(run_script, 'w') as r:
        for k in all_swan_files:
            r.write(f'swanrun -input {k}\n')

    os.chdir(TODAYSDATE)
    subprocess.call('bash run_all.sh', shell=True)
    os.chdir(original_dir)


def plot_wave_height_and_period(vtu_file):
    mesh = meshio.read(vtu_file)

    points = mesh.points
    x, y = points[:, 0], points[:, 1]
    triangles = mesh.cells_dict["triangle"]

    hs_meters = mesh.point_data["Hsig"]
    hs_feet = hs_meters * 3.28084
    wdir_deg = mesh.point_data["PkDir"]
    tpsmoo = mesh.point_data["Tm01"]
    depth = mesh.point_data["Depth"]

    triang = tri.Triangulation(x, y, triangles)

    colors_hs = [
        "#000080", "#0033cc", "#0066ff", "#00ccff", "#00ffff",
        "#00ff99", "#00ff00", "#66ff00", "#ccff00", "#ffff00",
        "#ffcc00", "#ff9900", "#ff6600", "#ff0000", "#cc0000",
    ]
    cmap_hs = ListedColormap(colors_hs)
    levels_hs = np.arange(0, 7.0, 0.5)
    norm_hs = BoundaryNorm(levels_hs, cmap_hs.N)

    base_cmap_tp = plt.colormaps['plasma']
    levels_tp = np.arange(5, 10.0, 0.2)
    cmap_tp = base_cmap_tp.resampled(len(levels_tp) - 1)
    norm_tp = BoundaryNorm(levels_tp, cmap_tp.N)

    bathy_levels = [5.0, 20.0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11), facecolor="lightgray")

    cf1 = ax1.tricontourf(triang, hs_feet, levels=levels_hs, cmap=cmap_hs, norm=norm_hs, extend="max")
    ax1.tricontour(triang, hs_feet, levels=np.arange(0, 7.5, 1.0), colors="k", linewidths=0.3)

    contour_bathy1 = ax1.tricontour(triang, depth, levels=bathy_levels, colors="#555555", linestyles="dashed", linewidths=0.6)
    ax1.clabel(contour_bathy1, inline=True, fmt="%d m", fontsize=8, colors="#555555")

    rad = np.deg2rad(wdir_deg)
    u, v = np.cos(rad), np.sin(rad)
    stride = 100000
    mask = hs_feet > 0.05

    ax1.quiver(
        x[mask][::stride], y[mask][::stride],
        u[mask][::stride], v[mask][::stride],
        color="white", edgecolor="black", linewidth=0.6,
        scale=35, width=0.0035, headwidth=4, pivot="middle"
    )

    ax1.set_aspect("equal")
    ax1.set_title("Significant Wave Height (ft)", fontsize=14, fontweight="bold")
    cbar1 = fig.colorbar(cf1, ax=ax1, orientation="horizontal", pad=0.06, fraction=0.045)
    cbar1.set_label("Wave Height (ft)", fontsize=11)

    cf2 = ax2.tricontourf(triang, tpsmoo, levels=levels_tp, cmap=cmap_tp, norm=norm_tp, extend="both")
    ax2.tricontour(triang, tpsmoo, levels=np.arange(5, 11.1, 1.0), colors="k", linewidths=0.4, alpha=0.7)

    contour_bathy2 = ax2.tricontour(triang, depth, levels=bathy_levels, colors="#444444", linestyles="dashed", linewidths=0.6)
    ax2.clabel(contour_bathy2, inline=True, fmt="%d m", fontsize=8, colors="#444444")

    ax2.set_aspect("equal")
    ax2.set_title("Mean Wave Period (s)", fontsize=14, fontweight="bold")
    cbar2 = fig.colorbar(cf2, ax=ax2, orientation="horizontal", pad=0.06, fraction=0.045, ticks=np.arange(5, 11.1, 1.0), format="%.1f")
    cbar2.set_label("Wave Period (seconds)", fontsize=11)

    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', which='major', labelsize=9, colors='#333333')
        ax.get_xaxis().get_major_formatter().set_scientific(False)
        ax.get_yaxis().get_major_formatter().set_scientific(False)
        ax.grid(True, linestyle="--", alpha=0.5, color="gray", linewidth=0.5)

    fig.suptitle(
        f"Maldives Experimental Wave Forecast System (MEWFS) Output\n{os.path.basename(vtu_file)}",
        fontsize=18, fontweight="bold", y=0.96
    )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    png_name = os.path.splitext(vtu_file)[0] + "_combined.png"
    plt.savefig(png_name, dpi=300, bbox_inches="tight", facecolor="lightgray")
    plt.close()

    print("Saved combined plot with bathymetry:", png_name)


def make_height_and_period():
    vtu_files = glob.glob(os.path.join(TODAYSDATE, '*.vtu'))
    for vtu in vtu_files:
        plot_wave_height_and_period(vtu)


def copy_pngs_to_web_folder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    destination = os.path.join(script_dir, "raalhu_web", "images", "Laamu", TODAYSDATE)
    os.makedirs(destination, exist_ok=True)

    png_files = glob.glob(os.path.join(TODAYSDATE, '*.png'))
    print(f"Found {len(png_files)} PNG files")

    for png in png_files:
        shutil.copy2(png, destination)
        print(f"Copied {os.path.basename(png)}")

    print(f"Files copied to {destination}")


def run_git_upload():
    original_dir = os.getcwd()
    target_repo = "raalhu_web" if os.path.exists("raalhu_web") else "raalhu-web"
    os.chdir(target_repo)
    
    try:
        # Pull latest remote changes first
        subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=True)
        # Execute the upload script
        subprocess.call('bash git_upload.sh', shell=True)
    except Exception as e:
        print(f"Git sync failed: {e}")
    finally:
        os.chdir(original_dir)

def run_sequence():
    global TODAYSDATE
    TODAYSDATE = arrow.now().format('YYYYMMDD')  # Update date daily dynamically
    
    send_startup_email()
    get_gfs_grib()
    get_bund_valss()
    get_winds()
    write_run_file()
    run_files()
    make_height_and_period()
    copy_pngs_to_web_folder()
    run_git_upload()
    send_conclude_email()


if __name__ == "__main__":
    schedule.every().day.at("10:30").do(run_sequence)

    print("MEWFS Scheduler Started. Waiting for 10:30...")
    while True:
        schedule.run_pending()
        time.sleep(1)
