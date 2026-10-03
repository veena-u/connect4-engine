
import argparse
import math
import queue
import random
import threading
import tkinter as tk

import torch

import connect4_az
from network import AlphaZeroNet, evaluate

CELL = 90
PAD = 14
HEADER = 64
WIDTH, HEIGHT = 7, 6
BOARD_W, BOARD_H = WIDTH * CELL, HEIGHT * CELL
CUT = 26  # corner-clip size on the board panel (smoothed into a soft rounded edge) 

ORIGIN_X = PAD
ORIGIN_Y = PAD + HEADER

BG = "#15141f"
INK = "#05050a"
PANEL_FILL = "#352a68"
PANEL_GLOW_A = "#c1121f"  # red chromatic fringe
PANEL_GLOW_B = "#669bbc"  # blue chromatic fringe
HOLE_FILL = "#16111f"
HOLE_RIM = "#7a65b8"

PIECE = {
    "x": dict(base="#c1121f", dark="#780000", shard="#fff0f6", ghost="#669bbc"),
    "o": dict(base="#669bbc", dark="#003049", shard="#eafeff", ghost="#c1121f"),
}

DROP_MS = 320
FPS_MS = 16
FONT = ("Impact", 22)
FONT_BIG = ("Impact", 44)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", help="path to a checkpoint.pt written by train.py")
    p.add_argument("--simulations", type=int, default=800, help="network's search budget per move")
    p.add_argument("--c-puct", type=float, default=1.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--second", action="store_true", help="let the network move first instead of you")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def load_net(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    shape, actions, players = connect4_az.OBSERVATION_SHAPE, connect4_az.NUM_ACTIONS, connect4_az.NUM_PLAYERS
    net_args = checkpoint["args"]
    net = AlphaZeroNet(shape, actions, players, net_args["channels"], net_args["blocks"]).to(device)
    net.load_state_dict(checkpoint["net"])
    net.eval()
    return net, checkpoint["iteration"]


def cell_center(row, col):
    return col * CELL + CELL / 2, row * CELL + CELL / 2


def diff_cell(old_lines, new_lines):
    """Finds the one (row, col) that changed, and what piece landed there."""
    for row, (old_line, new_line) in enumerate(zip(old_lines, new_lines)):
        for col, (old_cell, new_cell) in enumerate(zip(old_line, new_line)):
            if old_cell != new_cell:
                return row, col, new_cell
    return None


def clipped_rect(x0, y0, x1, y1, cut):
    """An octagon-ish rectangle with clipped corners, for an angular comic-panel look."""
    return [
        x0 + cut, y0, x1 - cut, y0, x1, y0 + cut, x1, y1 - cut,
        x1 - cut, y1, x0 + cut, y1, x0, y1 - cut, x0, y0 + cut,
    ]


def starburst_points(cx, cy, outer_r, inner_r, spikes, rotation=0.0):
    """A jagged comic-burst polygon, like a classic 'KAPOW' shape."""
    points = []
    for i in range(spikes * 2):
        r = outer_r if i % 2 == 0 else inner_r
        theta = rotation + math.pi * i / spikes
        points.extend([cx + r * math.cos(theta), cy + r * math.sin(theta)])
    return points


def draw_glitch_text(canvas, x, y, text, tag, font=FONT, fill="#ffffff", glitch=True):
    """Bold comic lettering: a thick black ink outline, optionally with a chromatic split.

    The chromatic split reads as a fun comic/glitch effect, but it muddies legibility
    on text people need to read at a glance (whose turn it is), so `glitch=False`
    skips it and draws a plain, high-contrast outlined label instead.
    """
    for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2), (-2, 0), (2, 0), (0, -2), (0, 2)]:
        canvas.create_text(x + dx, y + dy, text=text, fill=INK, font=font, tags=tag)
    if glitch:
        canvas.create_text(x - 3, y, text=text, fill=PANEL_GLOW_B, font=font, tags=tag)
        canvas.create_text(x + 3, y, text=text, fill=PANEL_GLOW_A, font=font, tags=tag)
    canvas.create_text(x, y, text=text, fill=fill, font=font, tags=tag)


class App:
    def __init__(self, root, net, driver, device, human_seat):
        self.root, self.net, self.driver, self.device, self.human_seat = root, net, driver, device, human_seat
        self.results = queue.Queue()
        self.thinking = False
        self.animating = False
        self.board_lines = self.driver.board().splitlines()

        root.configure(bg=BG)
        self.canvas = tk.Canvas(root, width=BOARD_W + 2 * PAD, height=BOARD_H + ORIGIN_Y + PAD,
                                 bg=BG, highlightthickness=0)
        self.canvas.pack(padx=4, pady=4)
        self.canvas.bind("<Button-1>", self.on_click)

        self.draw_static()
        self.root.after(FPS_MS, self.poll)
        if self.driver.turn() != self.human_seat:
            self.start_network_move()

    # ---- drawing -----------------------------------------------------

    def draw_board_frame(self):
        c = self.canvas
        x0, y0, x1, y1 = ORIGIN_X, ORIGIN_Y, ORIGIN_X + BOARD_W, ORIGIN_Y + BOARD_H
        c.create_polygon(clipped_rect(x0, y0, x1, y1, CUT), fill=PANEL_FILL, outline=INK, width=5, smooth=True)

    def draw_hole(self, row, col):
        cx, cy = cell_center(row, col)
        r = CELL / 2 - 9
        x, y = ORIGIN_X + cx, ORIGIN_Y + cy
        self.canvas.create_oval(x - r - 3, y - r - 3, x + r + 3, y + r + 3, fill=INK, outline="")
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=HOLE_FILL, outline=HOLE_RIM, width=2)

    def draw_piece(self, cx, cy, symbol, tag=None):
        """One piece: chromatic ghost rim, bold-inked disc, halftone shadow, angular highlight shard."""
        colors = PIECE[symbol]
        r = CELL / 2 - 9
        x, y = ORIGIN_X + cx, ORIGIN_Y + cy
        kwargs = {"tags": tag} if tag else {}

        # Same offset direction for every piece, regardless of color, so the implied
        # light source is consistent across the whole board instead of flipping per side.
        gx, gy = 3, 3
        self.canvas.create_oval(x - r + gx, y - r + gy, x + r + gx, y + r + gy,
                                 fill="", outline=colors["ghost"], width=2, **kwargs)
        self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                 fill=colors["base"], outline=INK, width=3, **kwargs)
        # Halftone-dot shadow instead of a smooth gradient.
        self.canvas.create_arc(x - r, y - r, x + r, y + r, start=250, extent=150,
                                fill=colors["dark"], outline="", stipple="gray50", style=tk.PIESLICE, **kwargs)
        # A sharp, angular cel-shaded highlight shard instead of a soft gloss oval.
        shard = [x - r * 0.5, y - r * 0.55, x - r * 0.05, y - r * 0.7, x - r * 0.25, y - r * 0.1]
        self.canvas.create_polygon(shard, fill=colors["shard"], outline="", **kwargs)

    def draw_static(self):
        self.canvas.delete("all")
        self.draw_board_frame()
        for row, line in enumerate(self.board_lines):
            for col, symbol in enumerate(line):
                self.draw_hole(row, col)
                if symbol != ".":
                    self.draw_piece(*cell_center(row, col), symbol)
        self.draw_status()

    def status_text(self):
        if self.driver.turn() == -1:
            winner = self.driver.winner()
            if winner is None:
                return "DRAW"
            return "YOU WIN!" if winner == self.human_seat else "NETWORK WINS"
        if self.thinking:
            return "NETWORK IS THINKING..."
        return "YOUR MOVE"

    def draw_status(self):
        self.canvas.delete("status")
        cx = ORIGIN_X + BOARD_W / 2
        text = self.status_text()
        if text == "YOUR MOVE":
            fill = "#ff4d4d"  # bright, high-contrast red; no chromatic split, for max legibility
        elif self.driver.turn() != -1:
            mover = self.driver.turn()
            fill = PIECE["x" if mover == 0 else "o"]["base"]
        else:
            fill = "#ffffff"
        draw_glitch_text(self.canvas, cx, HEADER / 2 + 6, text, tag="status", font=FONT, fill=fill, glitch=False)

    # ---- animation -----------------------------------------------------

    def animate_drop(self, row, col, symbol, on_done):
        self.animating = True
        cx, target_cy = cell_center(row, col)
        start_cy = -CELL / 2
        start = self.root.tk.call("clock", "milliseconds")

        def ease_in(t):
            return t * t  # accelerate, like gravity

        def step():
            now = self.root.tk.call("clock", "milliseconds")
            t = min(1.0, (now - start) / DROP_MS)
            cy = start_cy + (target_cy - start_cy) * ease_in(t)
            self.canvas.delete("falling")
            # Speed-line afterimages trailing above the piece, comic-motion style.
            for back in (26, 14):
                self.canvas.create_oval(
                    ORIGIN_X + cx - (CELL / 2 - 9) * 0.8, ORIGIN_Y + cy - back - 4,
                    ORIGIN_X + cx + (CELL / 2 - 9) * 0.8, ORIGIN_Y + cy - back + 4,
                    fill=PIECE[symbol]["ghost"], outline="", stipple="gray25", tags="falling",
                )
            self.draw_piece(cx, cy, symbol, tag="falling")
            if t < 1.0:
                self.root.after(FPS_MS, step)
            else:
                self.canvas.delete("falling")
                self.animating = False
                self.draw_static()
                on_done()

        step()

    # ---- turns -----------------------------------------------------

    def on_click(self, event):
        if self.animating or self.thinking or self.driver.turn() != self.human_seat:
            return
        col = (event.x - ORIGIN_X) // CELL
        if col not in self.driver.legal_moves():
            return
        old_lines = self.board_lines
        self.driver.apply_human_move(col)
        self.board_lines = self.driver.board().splitlines()
        row, col, symbol = diff_cell(old_lines, self.board_lines)
        self.animate_drop(row, col, symbol, self.after_move_landed)

    def start_network_move(self):
        self.thinking = True
        self.draw_status()
        threading.Thread(target=self.network_move, daemon=True).start()

    def network_move(self):
        old_lines = self.board_lines
        while not self.driver.is_thinking_done():
            observations, masks = self.driver.pending()
            self.driver.submit(*evaluate(self.net, observations, masks, self.device))
        self.driver.apply_network_move()
        new_lines = self.driver.board().splitlines()
        self.results.put((old_lines, new_lines))

    def poll(self):
        try:
            old_lines, new_lines = self.results.get_nowait()
            self.thinking = False
            self.board_lines = new_lines
            row, col, symbol = diff_cell(old_lines, new_lines)
            self.animate_drop(row, col, symbol, self.after_move_landed)
        except queue.Empty:
            pass
        self.root.after(FPS_MS, self.poll)

    def after_move_landed(self):
        if self.driver.turn() == -1:
            self.start_win_effect()
        elif self.driver.turn() != self.human_seat:
            self.start_network_move()

    # ---- end-of-game effects -----------------------------------------------------

    def start_win_effect(self):
        self.draw_status()
        winner = self.driver.winner()
        if winner == self.human_seat:
            self.crown_spin(angle=0.0)
        elif winner is not None:
            self.explosion(start=self.root.tk.call("clock", "milliseconds"))

    def crown_spin(self, angle):
        """A gold crown that spins forever above the board, comic-inked with a chromatic fringe."""
        self.canvas.delete("crown")
        cx, cy = ORIGIN_X + BOARD_W / 2, ORIGIN_Y - 26
        scale = math.cos(angle)
        bob = math.sin(angle * 2) * 4

        points = [(-70, 30), (-70, -10), (-35, 20), (0, -45), (35, 20), (70, -10), (70, 30)]
        poly = [coord for px, py in points for coord in (cx + px * scale, cy + py + bob)]
        self.canvas.create_polygon([c + 3 for c in poly], fill="", outline=PANEL_GLOW_B, width=2, tags="crown")
        self.canvas.create_polygon(poly, fill="#ffd700", outline=INK, width=3, tags="crown")

        for px, color in [(-35, "#c1121f"), (0, "#669bbc"), (35, "#b14bff")]:
            jx, jy = cx + px * scale, cy - 5 + bob
            self.canvas.create_oval(jx - 6, jy - 6, jx + 6, jy + 6, fill=color, outline=INK, width=1, tags="crown")

        if random.random() < 0.1:
            bx, by = cx + random.uniform(-90, 90), cy + random.uniform(-55, 15)
            bolt = starburst_points(bx, by, 10, 4, 4, rotation=random.uniform(0, math.pi))
            self.canvas.create_polygon(bolt, fill="#eafeff", outline="", tags="crown")

        self.root.after(FPS_MS, self.crown_spin, angle + 0.07)

    def explosion(self, start, particles=None):
        """An over-the-top glitch/comic explosion: shake, debris, a burst, chromatic strobe text."""
        now = self.root.tk.call("clock", "milliseconds")
        t = now - start
        duration = 1600

        if particles is None:
            cx, cy = ORIGIN_X + BOARD_W / 2, ORIGIN_Y + BOARD_H / 2
            colors = [PANEL_GLOW_A, PANEL_GLOW_B, "#b14bff", "#ffffff"]
            particles = []
            for _ in range(42):
                theta = random.uniform(0, 2 * math.pi)
                speed = random.uniform(2, 7)
                particles.append({
                    "x": cx, "y": cy,
                    "dx": math.cos(theta) * speed, "dy": math.sin(theta) * speed,
                    "color": random.choice(colors), "size": random.uniform(6, 15),
                })

        shake = max(0, 10 - t // 20)
        ox, oy = random.uniform(-shake, shake), random.uniform(-shake, shake)

        self.canvas.delete("fx")
        self.draw_static()
        self.canvas.move("all", ox, oy)
        self.canvas.addtag_withtag("fx", "all")

        cx, cy = ORIGIN_X + BOARD_W / 2 + ox, ORIGIN_Y + BOARD_H / 2 + oy
        burst = starburst_points(cx, cy, 170, 95, 9, rotation=t * 0.004)
        self.canvas.create_polygon(burst, fill="#2a1442", outline=PANEL_GLOW_A, width=3, tags="fx")

        for p in particles:
            p["x"] += p["dx"] + ox
            p["y"] += p["dy"] + oy
            p["dy"] += 0.15
            p["size"] *= 0.97
            if p["size"] > 1:
                r = p["size"]
                self.canvas.create_rectangle(p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r,
                                              fill=p["color"], outline=INK, tags="fx")

        if t < duration * 0.8:
            draw_glitch_text(self.canvas, cx, cy, "DANG IT!", tag="fx", font=FONT_BIG)

        if t < duration:
            self.root.after(FPS_MS, self.explosion, start, particles)
        else:
            self.canvas.delete("fx")
            self.draw_static()


def main():
    args = parse_args()
    device = torch.device(args.device)
    net, iteration = load_net(args.checkpoint, device)
    driver = connect4_az.PlayDriver(args.simulations, args.c_puct, args.seed)
    human_seat = 1 if args.second else 0

    root = tk.Tk()
    root.title(f"Connect 4 vs. checkpoint (iteration {iteration})")
    root.resizable(False, False)
    App(root, net, driver, device, human_seat)
    root.mainloop()


if __name__ == "__main__":
    main()
