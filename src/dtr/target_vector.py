# ===== BEGIN Connor's solution: Vector_Creation (target_vector_creator.py) =====
"""
Target_vector_creation.py - goal pixel -> yaw change, elevation change, travel distance.

    yaw_deg, elev_change_m, travel_m = target_vector(x_pix_cord, y_pix_cord, goal_type, goal_height_px)

Inputs
    x_pix_cord, y_pix_cord  goal center pixel (x right, y down, 0,0 = top-left):
                            the MIDDLE of the goal's bounding box, so the goal's top is
                            half its height above y and its bottom half its height below
    goal_type               "square", "triangle", "circle" or "balloon"
    goal_height_px          goal height in pixels, measured down the goal's vertical
                            CENTER LINE (see "Measuring the height" below).
                            Needed: position alone can't tell near from far.

Returns
    yaw_deg        degrees to turn (+ right, - left)
    elev_change_m  meters to climb (+) or descend (-) to be level with the goal's
                   opening center (box middle minus AIM_BELOW_MIDDLE_M)
    travel_m       meters to travel forward once at the target yaw and elevation

Steps
    1. undistort: fisheye pixel -> flat-lens coordinates (direction to the goal)
    2. distance: find the depth at which the goal (known real height) would look
       goal_height_px tall through this lens
    3. offsets: right = x_flat * depth, up = -y_flat * depth (relative to the origin)
    4. yaw = atan2(right, depth), elevation change = up,
       travel = sqrt(right^2 + depth^2)

Measuring the height (center line, not the bounding box)
    find_depth() assumes goal_height_px is measured straight down the target's
    vertical center line: from its top to its bottom through its middle x. For a
    target measured that way the result is exact.
    If you pass the full BOUNDING BOX height instead, targets near the edges and
    corners of the image read CLOSER than they are, because the fisheye bends the
    target and its box comes out taller than its center line. Near the image center
    there is almost no difference; near a corner it can be ~14% short (e.g. a square
    at pixel (98, 786), 40 px tall: 11.68 m from the center line vs 13.61 m when the
    box bending is accounted for).
    The box is also too tall when the target is turned away from the camera (seen
    from the side): its near edge is closer, looks taller, and sets the box height,
    so the distance comes out as roughly the distance to the near edge.
    Fix: have the goal finder measure the height in the target's center column
    (first target pixel from the top and last from the bottom at the middle x).
    That matches what this script expects and removes both errors.

Assumes a level camera (no roll/pitch) and an upright goal. Pure Python.
"""
import math


class VectorCreator:
    # ADAPTED: calibration and dimensions come from the runtime lens.
    def __init__(self, lens, sizes, origin=None, pass_through_margin_m=0.0,
                 triangle_point_down=True):
        self.IMAGE_WIDTH, self.IMAGE_HEIGHT = lens.width, lens.height
        self.FX, self.FY = lens.K[0, 0], lens.K[1, 1]
        self.CX, self.CY = lens.K[0, 2], lens.K[1, 2]
        self.DIST = lens.D.ravel().tolist()
        self.K1, self.K2, self.K3, self.K4 = self.DIST
        self.ORIGIN_X, self.ORIGIN_Y = origin or (None, None)
        self.PASS_THROUGH_MARGIN_M = pass_through_margin_m
        self.sizes = sizes
        # ADAPTED: use rulebook outer sizes; Connor used square .6096,
        # circle .9271 (inside diameter), triangle 1.5146, balloon .762.
        self.TARGET_HEIGHT_M = {label: s.get('height_m', s['width_m'])
                                for label, s in sizes.items()}
        for label in sizes:
            self.TARGET_HEIGHT_M.setdefault(label.split('_')[-1], self.TARGET_HEIGHT_M[label])
        # ADAPTED: point-down centroid is H/6 above the box middle.
        self.AIM_BELOW_MIDDLE_M = {
            label: (-1 if triangle_point_down else 1) * h / 6
            if label.endswith('triangle') else 0.0
            for label, h in self.TARGET_HEIGHT_M.items()}

    # ADAPTED: flight-loop wrapper; box middle + centre-line height in, None when unusable.
    def vector_for(self, label, box, center_height_px=None):
        h = box[3] - box[1] if center_height_px is None else center_height_px
        try:
            result = self.target_vector((box[0]+box[2])/2, (box[1]+box[3])/2, label, h)
            if not all(math.isfinite(v) for v in result):
                return None
            return dict(zip(('yaw_deg', 'elev_m', 'travel_m'), result))
        except (ValueError, KeyError, ZeroDivisionError, OverflowError):
            return None

    def _bend(self, x, y):
        """Flat-lens coordinates -> fisheye coordinates (both in units of f)."""
        r = math.hypot(x, y)
        if r < 1e-12:
            return x, y
        t = math.atan(r)
        t2 = t * t
        s = t * (1 + t2 * (self.K1 + t2 * (self.K2 + t2 * (self.K3 + t2 * self.K4)))) / r
        return x * s, y * s


    def flat_to_pixel(self, x, y):
        """Flat-lens coordinates -> pixel."""
        xb, yb = self._bend(x, y)
        return self.FX * xb + self.CX, self.FY * yb + self.CY


    def pixel_to_flat(self, u, v):
        """Pixel -> flat-lens coordinates (undistort). (x, y, 1) is the direction of the
        light that hit this pixel: x right, y down, per 1 unit forward."""
        xb, yb = (u - self.CX) / self.FX, (v - self.CY) / self.FY
        rb = math.hypot(xb, yb)                     # fisheye radius (= angle if all k = 0)
        if rb < 1e-12:
            return xb, yb
        t = rb
        for _ in range(20):                         # Newton: solve for the true angle t
            t2 = t * t
            f = t * (1 + t2 * (self.K1 + t2 * (self.K2 + t2 * (self.K3 + t2 * self.K4)))) - rb
            df = 1 + t2 * (3 * self.K1 + t2 * (5 * self.K2 + t2 * (7 * self.K3 + 9 * t2 * self.K4)))
            step = f / df
            t -= step
            if abs(step) < 1e-12:
                break
        if not 0 <= t < math.pi / 2:
            raise ValueError("pixel is outside the usable lens range")
        s = math.tan(t) / rb                        # angle -> flat-lens radius
        return xb * s, yb * s


    # ---------- distance ----------
    def find_depth(self, xf, yf, goal_type, goal_height_px):
        """Forward depth (m) at which the target, centered on direction (xf, yf), would look
        goal_height_px tall in this image. Its top is half its height above the middle and
        its bottom half its height below, on the vertical line through the middle.
        goal_height_px must be measured on that center line, not the bounding box
        (box heights read short near the image edges; see the top of the file)."""
        height = self.TARGET_HEIGHT_M[goal_type]
        half = height / 2                                # top/bottom are half above/below
        depth = self.FY * height / goal_height_px             # flat-lens start guess
        for _ in range(30):              # looks ~1/depth, so rescale until it matches
            top = self.flat_to_pixel(xf, yf - half / depth)[1]
            bot = self.flat_to_pixel(xf, yf + half / depth)[1]
            new = depth * (bot - top) / goal_height_px
            if abs(new - depth) < 1e-9 * depth:
                return new
            depth = new
        return depth

    #############################################
    #### ---------- main function ---------- ####
    #############################################
    def target_vector(self, x_pix_cord, y_pix_cord, goal_type, goal_height_px):
        if goal_type not in self.TARGET_HEIGHT_M:
            raise ValueError(f"unknown goal_type {goal_type!r}, use one of {list(self.TARGET_HEIGHT_M)}")
        if goal_height_px <= 0:
            raise ValueError("goal_height_px must be > 0")
        ox = self.IMAGE_WIDTH / 2 if self.ORIGIN_X is None else self.ORIGIN_X
        oy = self.IMAGE_HEIGHT / 2 if self.ORIGIN_Y is None else self.ORIGIN_Y

        # 1) undistort to flat-lens coordinates
        xf, yf = self.pixel_to_flat(x_pix_cord, y_pix_cord)     # goal direction
        xo, yo = self.pixel_to_flat(ox, oy)                     # heading direction

        # 2) how far ahead the goal is
        depth = self.find_depth(xf, yf, goal_type, goal_height_px)

        # 3) offsets in meters
        right = xf * depth                                 # right of the camera axis
        up = (yo - yf) * depth                             # box middle, above the heading

        # 4) outputs
        yaw_deg = math.degrees(math.atan2(right, depth) - math.atan(xo))
        elev_change_m = up - self.AIM_BELOW_MIDDLE_M[goal_type]
        travel_m = math.hypot(right, depth) + self.PASS_THROUGH_MARGIN_M
        return yaw_deg, elev_change_m, travel_m
# ===== END Connor's solution =====
