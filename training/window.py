import pygetwindow as gw

def focus_gd_window():
    windows = gw.getWindowsWithTitle("Geometry Dash")
    if windows:
        windows[0].activate()
        print("🌟 Окно GD активировано")
