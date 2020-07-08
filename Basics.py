import math
import numpy as np
import matplotlib.pyplot as plt
import pygame
import sys
import os
from PIL import Image


def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


ImageFolder = resource_path('images')


def map_one(x, y):
    return (40 * math.exp(-(((x - 450) - 0.001 * y ** 2) / 150) ** 2)
            + 30 * math.exp(-(x / 300) ** 2 - (y / 300) ** 2)
            + 15 * math.cos(x / 200) * math.sin(-y / 200) * math.exp(-(x + 100) / 500))


def map_two(x, y):
    return (15 * math.exp(-((x / 400) ** 2 + (y / 200) ** 2)) * math.cos((x / 100) ** 2 + (y / 100) ** 2)
            + 5 * math.exp(-((x / 50) ** 2 + ((y + 400) / 50) ** 2))
            + line_seg([800, 450], [400, 950], 40, 100)(x, y)
            + circ(690, 560, 20, 70)(x, y))


def map_three(x, y):
    return (10 * math.sin(x / 75) * math.sin(y / 75)) - y / 15


def circ(a, b, bh, bt):
    return lambda x, y: bh * math.exp(-((x - (a - 800)) / bt) ** 2 - ((y - (-b + 450)) / bt) ** 2)


def line_seg(A, B, bh, bt):
    a, b = [A[0] - 800, -A[1] + 450], [B[0] - 800, - B[1] + 450]
    try:
        m = (b[1] - a[1]) / (b[0] - a[0])
    except ZeroDivisionError:
        m = (a[1] - b[1]) * 10 ** 100
    try:
        m_prime = -1 / m
    except ZeroDivisionError:
        m_prime = 10 ** 100

    def line_seg_r(x, y):
        if a[1] < b[1]:
            if y < a[1] + m_prime * (x - a[0]):
                return bh * math.exp(- ((x - a[0]) ** 2 + (y - a[1]) ** 2) / ((bt ** 2) / (m ** 2 + 1)))
            if a[1] + m_prime * (x - a[0]) <= y <= b[1] + m_prime * (x - b[0]):
                return bh * math.exp(- ((y - a[1] - m * (x - a[0])) / bt) ** 2)
            if y > b[1] + m_prime * (x - b[0]):
                return bh * math.exp(- ((x - b[0]) ** 2 + (y - b[1]) ** 2) / ((bt ** 2) / (m ** 2 + 1)))
        elif a[1] > b[1]:
            if y > a[1] + m_prime * (x - a[0]):
                return bh * math.exp(- ((x - a[0]) ** 2 + (y - a[1]) ** 2) / ((bt ** 2) / (m ** 2 + 1)))
            if a[1] + m_prime * (x - a[0]) >= y >= b[1] + m_prime * (x - b[0]):
                return bh * math.exp(- ((y - a[1] - m * (x - a[0])) / bt) ** 2)
            if y < b[1] + m_prime * (x - b[0]):
                return bh * math.exp(- ((x - b[0]) ** 2 + (y - b[1]) ** 2) / ((bt ** 2) / (m ** 2 + 1)))

    return lambda x, y: line_seg_r(x, y)


def height_map(map, bar):
    x, y = np.meshgrid(np.linspace(-800, 800, 1600), np.linspace(-450, 450, 900))
    z = np.array([[map(x, y) for x in range(-800, 800)] for y in range(-450, 450)])
    z_min, z_max = -np.abs(z).max(), np.abs(z).max()
    fig, ax = plt.subplots()
    c = ax.pcolormesh(x, y, z, cmap='viridis', vmin=z_min, vmax=z_max)
    # ax.set_title('Height')
    fig.set_size_inches(20.8, 11.7)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    plt.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    if bar:
        c.set_visible(False)
        cb = fig.colorbar(c, ax=ax, orientation="vertical")
        cb.ax.tick_params(labelsize=14)
        cbytick_obj = plt.getp(cb.ax.axes, 'yticklabels')
        plt.setp(cbytick_obj, color='black', weight='bold')
        fig.set_size_inches(3.5, 11.7)
        plt.savefig(os.path.join(ImageFolder, 'color_bar.png'), bbox_inches='tight', pad_inches=0, dpi=100)
    else:
        plt.savefig(os.path.join(ImageFolder, 'height_map.png'), bbox_inches='tight', pad_inches=0, dpi=100)


def remove_background(im_in):
    img = Image.open(im_in)
    img = img.convert("RGBA")
    data = img.getdata()
    newData = []
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    img.save(os.path.join(ImageFolder, 'color_bar.png'), "PNG")


def generate_map(map):
    height_map(map, False)
    height_map(map, True)
    remove_background(os.path.join(ImageFolder, "color_bar.png"))


def visual_map(map, scaled):
    z = np.array([[map(x, y) for x in range(-80, 80, 2)] for y in range(0, 100)])
    x, y = np.meshgrid(np.linspace(-80, 80, 80), np.linspace(0, 100, 100))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if scaled:
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.plot_surface(x, y, z)
    plt.title('Map Model')
    plt.show()


class Background(pygame.sprite.Sprite):
    def __init__(self, image_file, location):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(os.path.join(ImageFolder, image_file))
        self.rect = self.image.get_rect()
        self.rect.left, self.rect.top = location


class Map:
    def __init__(self, map, mu, hole):
        self.h = map
        self.mu = mu
        self.hole = hole
        self.name = str(map)

    def grad(self, pos):
        grad_x = (self.h(pos[0] + 0.0001, pos[1]) - self.h(pos[0] - 0.0001, pos[1])) / 0.0002
        grad_y = (self.h(pos[0], pos[1] + 0.0001) - self.h(pos[0], pos[1] - 0.0001)) / 0.0002
        return np.array([grad_x, grad_y])

    def weight(self, pos):
        g = 9.81
        acc_x = - g * (Map.grad(self, pos)[0]) / ((1 + (Map.grad(self, pos)[0]) ** 2) ** 0.5)
        acc_y = - g * (Map.grad(self, pos)[1]) / ((1 + (Map.grad(self, pos)[1]) ** 2) ** 0.5)
        return np.array([acc_x, acc_y])

    def fric_mag(self, pos, vel):
        g = 9.81
        vel_dir = vel / np.linalg.norm(vel)
        dir_grad = np.dot(vel_dir, Map.grad(self, pos))
        return g * self.mu / (1 + dir_grad ** 2) ** 0.5

    def win_spot(self, pos):
        return (pos[0] - self.hole[0]) ** 2 + (pos[1] - self.hole[1]) ** 2 < 225

    def static_fric_winning(self, pos):
        return ((abs(1.1 * self.mu) > abs(Map.grad(self, pos)[0]))
                and (abs(1.1 * self.mu) > abs(Map.grad(self, pos)[1])))

    def max(self):
        return max(list(self.h(x, y) for x in range(-800, 800) for y in range(-450, 450)))


class Ball:
    def __init__(self, x, y, angle, radius, mass):
        self.pos = np.array([x, y])
        self.vel = np.array([float(0), float(0)])
        self.ang = angle
        self.rad = radius
        self.m = mass

    def hit(self, ang, v_in):
        v_x = v_in * math.sin(ang)
        v_y = v_in * math.cos(ang)
        self.vel = np.array([v_x, v_y])

    def set(self, pos, ang, v_in):
        v_x = v_in * math.sin(ang)
        v_y = v_in * math.cos(ang)
        self.pos = pos
        self.ang = ang
        self.vel = np.array([v_x, v_y])

    def update(self, map, dt):
        weight = map.weight(self.pos)
        fric_mag = map.fric_mag(self.pos, self.vel)
        self.pos = self.pos + self.vel * dt + 0.5 * weight * dt ** 2
        self.vel = self.vel + weight * dt
        if fric_mag * dt / np.linalg.norm(self.vel) < 1:
            self.vel = self.vel * (1 - fric_mag * dt / np.linalg.norm(self.vel))
        else:
            self.vel = np.array([0, 0])
        try:
            if self.vel[1] > 0:
                self.ang = math.atan(float(self.vel[0]) / float(self.vel[1]))
            elif self.vel[1] < 0 and (self.vel[0] > 0):
                self.ang = math.pi + math.atan(float(self.vel[0]) / float(self.vel[1]))
            elif self.vel[1] < 0 and (self.vel[0] < 0):
                self.ang = -math.pi + math.atan(float(self.vel[0]) / float(self.vel[1]))
        except ZeroDivisionError:
            self.ang = math.atan(float(self.vel[0]) * 10 ** 100)


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Golf")
        width = 1600
        height = 900
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.ticks = 60
        self.exit = False

    def run(self):

        bg = Background('height_map1.png', [0, 0])
        ball_im = pygame.image.load(os.path.join(ImageFolder, 'ball2.png'))
        ind_im = pygame.image.load(os.path.join(ImageFolder,'indicator.png'))
        flag_im = pygame.image.load(os.path.join(ImageFolder,'flag.png'))
        color_im = pygame.image.load(os.path.join(ImageFolder,'color_bar1.png'))

        ball = Ball(0, -400, 0, 16, 1)  # Initial x, Initial y, Initial Angle, Radius, Mass
        hole = np.array([200, 375])
        map = Map(map_one, 0.1, hole)  # Height_Func, Friction Coefficient, Position of Hole
        max_h = 46.690902202783946

        hit_ang = 0
        hit_v = 32
        vision = 1
        mu = 0.1

        best_dist = ((ball.pos[0] - hole[0]) ** 2 + (ball.pos[1] - hole[1]) ** 2) ** 0.5
        dist = ((ball.pos[0] - hole[0]) ** 2 + (ball.pos[1] - hole[1]) ** 2) ** 0.5

        live = False
        ready = True
        show_guide = True
        win = False
        click = False
        editor = False
        update_map = False
        switch_map = False

        map_new = map_one
        hole_new = np.array([200, 375])

        mode = "mouse"
        brush_thick = 100
        brush_height = 10
        funcs = []

        prevtype = "circle"

        centers = []
        centers_draw = []
        prev = None
        prev2 = None

        end_point = 0
        line = []
        lines = []
        lines_draw = []
        l_prev1 = None
        l_prev2 = None

        visuals = "Shot"
        visuals_bool = True
        res = 3

        vision_ui = pygame.Rect(10, 10, 155, 40)
        vision_down = pygame.Rect(10, 56, 75, 40)
        vision_up = pygame.Rect(90, 56, 75, 40)
        mu_ui = pygame.Rect(10, 102, 155, 40)
        mu_down = pygame.Rect(10, 148, 75, 40)
        mu_up = pygame.Rect(90, 148, 75, 40)
        reset = pygame.Rect(10, 194, 155, 40)

        map_menu = pygame.Rect(10, 246, 155, 40)
        map_1 = pygame.Rect(10, 292, 48, 40)
        map_2 = pygame.Rect(63, 292, 49, 40)
        map_3 = pygame.Rect(117, 292, 48, 40)
        map_editor = pygame.Rect(10, 338, 155, 40)

        editor_exit = pygame.Rect(432, 850, 100, 40)
        flag_add = pygame.Rect(432, 805, 100, 40)

        map_circle = pygame.Rect(537, 805, 100, 40)
        map_line = pygame.Rect(537, 850, 100, 40)

        draw_thickness = pygame.Rect(642, 805, 155, 40)
        thickness_down = pygame.Rect(642, 850, 75, 40)
        thickness_up = pygame.Rect(722, 850, 75, 40)

        draw_height = pygame.Rect(802, 805, 155, 40)
        height_down = pygame.Rect(802, 850, 75, 40)
        height_up = pygame.Rect(882, 850, 75, 40)

        undo = pygame.Rect(962, 805, 100, 40)
        redo = pygame.Rect(962, 850, 100, 40)

        map_clear = pygame.Rect(1067, 805, 100, 40)
        update_key = pygame.Rect(1067, 850, 100, 40)

        solutions = pygame.Rect(10, 390, 155, 40)
        visualize = pygame.Rect(10, 436, 155, 40)
        angle_res = pygame.Rect(10, 482, 155, 40)
        res_down = pygame.Rect(10, 528, 75, 40)
        res_up = pygame.Rect(90, 528, 75, 40)
        bisect = pygame.Rect(10, 574, 155, 40)
        brute = pygame.Rect(10, 620, 155, 40)
        gradient = pygame.Rect(10, 666, 155, 40)
        nadam = pygame.Rect(10, 713, 155, 40)

        dist_bar = pygame.Rect(10, 760, 155, 40)
        angle_bar = pygame.Rect(10, 806, 155, 40)
        vel_bar = pygame.Rect(10, 852, 155, 40)

        game_font = pygame.font.SysFont('Arial', 20)

        def visual_path(ang, v):
            visual_ball = Ball(0, -400, 0, 5, 1)
            visual_ball.set(np.array([0, -400]), math.radians(ang), v)
            for i in range(30):
                if not (map.static_fric_winning(visual_ball.pos) and (
                        abs(visual_ball.vel) < 0.5 + 10 * mu).all()):
                    for j in range(10):
                        visual_ball.update(map, dt=dt_c)
                rotated_ind = pygame.transform.rotate(ind_im, math.degrees(-visual_ball.ang))
                self.screen.blit(rotated_ind,
                                 [int(visual_ball.pos[0]) + 800 - 8, -int(visual_ball.pos[1]) + 450 - 8])
                pygame.event.pump()
                blit()
                pygame.display.flip()

        def visual_shot(ang, v, a_count, v_count):
            disp = [v ** 0.5 * math.sin(ang), v ** 0.5 * math.cos(ang)]
            end_pos = [round(800 + 25 * disp[0]), round(850 - 25 * disp[1])]
            pygame.draw.line(self.screen, (round(27 * a_count) % 254, 25, 255 - round(24 * v_count) % 254),
                             [800, 850], end_pos, 3)
            pygame.event.pump()
            blit()
            pygame.display.flip()

        def angle_bisection(map, ball, ang_range, v_in, dt, v_count, a_count):
            mid = (ang_range[0] + ang_range[1]) / 2
            if visuals_bool:
                visual_shot(mid, v_in, a_count, v_count)
            elif not visuals_bool:
                visual_path(math.degrees(mid), v_in)
            best_measure = measure(map, ball, mid, v_in, dt, False)
            side = best_measure[1]
            dir_change = best_measure[2]
            y_dir = best_measure[3]
            if abs(dir_change) < 10:
                dir_change = 1
            direction = side * dir_change * y_dir
            if best_measure[0] < 600:
                if best_measure[0] == 0:
                    return math.degrees(mid)
                elif math.degrees(abs(ang_range[1] - ang_range[0])) < 0.5:
                    return "None"
                elif direction > 0:
                    new_ang_range = [ang_range[0], mid]
                    return angle_bisection(map, ball, new_ang_range, v_in, dt, v_count, a_count)
                elif direction < 0:
                    new_ang_range = [mid, ang_range[1]]
                    return angle_bisection(map, ball, new_ang_range, v_in, dt, v_count, a_count)
            else:
                return "None"

        def broad_angle_bisection(map, ball, v_in, n, dt, v_count):
            sub_size = math.radians(90) / n
            for i in range(n):
                ang_range = [i * sub_size, (i + 1) * sub_size]
                ans = angle_bisection(map, ball, ang_range, v_in, dt, v_count, (9 * i) / n)
                if type(ans) != str:
                    return ans
                ang_range = [- (i + 1) * sub_size, - i * sub_size]
                ans = angle_bisection(map, ball, ang_range, v_in, dt, v_count, (9 * i) / n)
                if type(ans) != str:
                    return ans
            return "None"

        def bisection_search(map, ball, n, v_range, dt, lowest, v_count):
            mid = (v_range[1] + v_range[0]) / 2
            angle = broad_angle_bisection(map, ball, mid, n, dt, v_count)
            v_count += 1
            # print(angle, mid)
            if abs(mid - lowest[1]) < 0.4:
                return lowest
            if type(angle) != str:
                if mid < lowest[1]:
                    lowest = angle, mid
                v_range_new = [v_range[0], mid]
                return bisection_search(map, ball, n, v_range_new, dt, lowest, v_count)
            if angle == "None":
                v_range_new = [mid, v_range[1]]
                return bisection_search(map, ball, n, v_range_new, dt, lowest, v_count)

        def min_vel(map, start):
            hole = map.hole
            dist = ((hole[0] - start[0]) ** 2 + (hole[1] - start[1]) ** 2) ** 0.5
            H_h = map.h(hole[0], hole[1])
            H_s = map.h(start[0], start[1])
            val = 2 * 9.81 * (H_h - H_s + (map.mu * dist) / (1 + (abs(H_h - H_s) / dist) ** 2) ** 0.5)
            if val > 0:
                return val ** 0.5
            else:
                return 0

        def smart_brute_search(map, ball, n, min, dt, lowest, v_count):
            v_count += 1
            if min <= 0:
                min = 1
            for v in range(round(min), round(lowest[1])):
                angle = broad_angle_bisection(map, ball, v, n, dt, v_count)
                v_count += 1
                # print(v, angle)
                if type(angle) != str:
                    return angle, v

        def one_grad_descent(f, x, y, mu, f_mu, n, count):
            ans = np.array([float(x), float(y)])
            step_c = 0.01
            iterations = 15

            def grad(f, x, y):
                grad_x = (f(x + 0.0001, y) - f(x - 0.0001, y)) / 0.0002
                grad_y = (f(x, y + 0.0001) - f(x, y - 0.0001)) / 0.0002
                return np.array([grad_x, grad_y])

            for i in range(iterations):
                new_val = None
                if mu <= 0.15:
                    grad_vect = grad(f, ans[0], ans[1])
                    func_val = f(ans[0], ans[1])
                    step = - step_c * (func_val - ans[1] ** 2 / 10 + 5) * (grad_vect / np.linalg.norm(grad_vect))
                    ans += np.array([10 / n * step[0], step[1]])
                    new_val = f(ans[0], ans[1])
                elif mu > 0.15:
                    grad_vect = grad(f_mu, ans[0], ans[1])
                    func_val = f_mu(ans[0], ans[1])
                    step = - step_c * (grad_vect / np.linalg.norm(grad_vect)) * (func_val + 50) / 5
                    ans += np.array([10 / n * step[0], step[1]])
                    new_val = f_mu(ans[0], ans[1])
                # print(ans)
                if not ((-90 <= ans[0] <= 90) and (0 <= ans[1] <= 100)):
                    break
                if not visuals_bool:
                    visual_path(ans[0], ans[1])
                elif visuals_bool:
                    visual_shot(math.radians(ans[0]), ans[1], i / 2, 10 * count)
                if mu <= 0.15:
                    if new_val < (ans[1] ** 2 / 10 + 15):
                        return ans
                elif mu > 0.15:
                    if new_val < 15:
                        return ans

            return "None"

        def full_grad_descent(map, ball, min, n, dt):
            sub_size = 90 / n
            if min == 0:
                min = 1
            for j in range(5):
                for i in range(n):
                    mid = (i * sub_size + (i + 1) * sub_size) / 2
                    ans = one_grad_descent(cost_func(map, ball, dt), mid, min + 15 * j, map.mu,
                                           cost_func_high_mu(map, ball, dt), n, i)
                    if type(ans) != str:
                        return ans
                    mid = (- (i + 1) * sub_size - i * sub_size) / 2
                    ans = one_grad_descent(cost_func(map, ball, dt), mid, min + 15 * j, map.mu,
                                           cost_func_high_mu(map, ball, dt), n, i)
                    if type(ans) != str:
                        return ans
            return "None"

        def nadam_grad_descent(f, x, y, mu, count):
            ans = np.array([float(x), float(y)])
            iterations = 40
            gamma = 0.75
            gamma2 = 0.9
            m_step = np.array([0, 0])
            n_step = np.array([0, 0])
            made_it = []

            def vect_mult(x, y):
                return np.array([x[0] * y[0], x[1] * y[1]])

            def grad(f, x, y):
                grad_x = (f(x + 0.0001, y) - f(x - 0.0001, y)) / 0.0002
                grad_y = (f(x, y + 0.0001) - f(x, y - 0.0001)) / 0.0002
                return np.array([grad_x, grad_y])

            for i in range(1, iterations):
                step_c = 3
                grad_vect = grad(f, ans[0], ans[1])
                m_step = gamma * m_step + (1 - gamma) * grad_vect
                n_step = gamma2 * n_step + (1 - gamma2) * vect_mult(grad_vect, grad_vect)
                m_fix = gamma * m_step / (1 - gamma ** (i + 1)) + (1 - gamma) * grad_vect / (1 - gamma ** i)
                n_fix = n_step / (1 - gamma2 ** i)
                ans = ans - step_c * m_fix / (n_fix ** 0.5 + 0.000001)
                new_val = f(ans[0], ans[1])
                # print(ans)
                if not ((-90 <= ans[0] <= 90) and (0 <= ans[1] <= 100)):
                    break
                if new_val < ans[1] ** (3 * (1 - mu / 1.5)) + 15 ** 2:
                    made_it.append(ans)
                if not visuals_bool:
                    visual_path(ans[0], ans[1])
                elif visuals_bool:
                    visual_shot(math.radians(ans[0]), ans[1], i / 4, 10 * count)
            return min(made_it, default="None", key=lambda elm: elm[1])

        def nadam_full_grad_descent(map, ball, min_vel, n, dt):
            sub_size = 90 / n
            if min_vel == 0:
                min_vel = 1
            answers = []
            for i in range(n):
                mid = (i * sub_size + (i + 1) * sub_size) / 2
                ans = nadam_grad_descent(nadam_cost_func(map, ball, dt), mid, min_vel, map.mu, i)
                # print(ans)
                if type(ans) != str:
                    return ans
                mid = (- (i + 1) * sub_size - i * sub_size) / 2
                ans = nadam_grad_descent(nadam_cost_func(map, ball, dt), mid, min_vel, map.mu, i)
                if type(ans) != str:
                    return ans
            return min(answers, default="None", key=lambda elm: elm[1])

        while not self.exit:
            # dt = self.clock.get_time() / 250
            dt_c = 0.4
            self.screen.fill([255, 255, 255])
            self.screen.blit(bg.image, bg.rect)

            mx, my = pygame.mouse.get_pos()

            if update_map:
                pygame.event.pump()
                generate_map(map_new)
                bg = Background('height_map.png', [0, 0])
                color_im = pygame.image.load(os.path.join(ImageFolder,'color_bar.png'))
                map = Map(map_new, mu, hole_new)
                hole = hole_new
                max_h = map.max()
                ball.set(np.array([0, -400]), hit_ang, hit_v)
                win = False
                live = False
                ready = False
                show_guide = False
                best_dist = ((ball.pos[0] - hole_new[0]) ** 2 + (ball.pos[1] - hole_new[1]) ** 2) ** 0.5

                update_map = False
                if not editor:
                    show_guide = True
                    ready = True

            if switch_map:
                map = Map(map_new, mu, hole_new)
                hole = hole_new
                win = False
                live = False
                ready = True
                show_guide = True
                ball.set(np.array([0, -400]), hit_ang, hit_v)
                best_dist = ((ball.pos[0] - hole_new[0]) ** 2 + (ball.pos[1] - hole_new[1]) ** 2) ** 0.5
                switch_map = False
                editor = False

            if vision_down.collidepoint(mx, my):
                if click:
                    if vision > 1:
                        vision -= 1
                    else:
                        vision = 1

            if vision_up.collidepoint(mx, my):
                if click:
                    if vision < 15:
                        vision += 1
                    else:
                        vision = 15

            if mu_down.collidepoint(mx, my):
                if click:
                    if mu > 0.04:
                        mu -= 0.025
                        map = Map(map_new, mu, hole)
                    else:
                        mu = 0.025

            if mu_up.collidepoint(mx, my):
                if click:
                    if mu < 0.5:
                        mu += 0.025
                        map = Map(map_new, mu, hole)
                    else:
                        mu = 0.5

            if reset.collidepoint(mx, my):
                if click:
                    ball.set(np.array([0, -400]), hit_ang, hit_v)
                    win = False
                    live = False
                    ready = True
                    show_guide = True
                    best_dist = ((ball.pos[0] - hole[0]) ** 2 + (ball.pos[1] - hole[1]) ** 2) ** 0.5

            if map_1.collidepoint(mx, my):
                if click:
                    map_new = map_one
                    hole_new = np.array([200, 375])
                    mu = 0.1
                    bg = Background('height_map1.png', [0, 0])
                    color_im = pygame.image.load(os.path.join(ImageFolder,'color_bar1.png'))
                    max_h = 46.690902202783946
                    switch_map = True

            if map_2.collidepoint(mx, my):
                if click:
                    map_new = map_two
                    hole_new = np.array([-175, 0])
                    mu = 0.075
                    bg = Background('height_map2.png', [0, 0])
                    color_im = pygame.image.load(os.path.join(ImageFolder,'color_bar2.png'))
                    max_h = 65.30873487977566
                    switch_map = True

            if map_3.collidepoint(mx, my):
                if click:
                    map_new = map_three
                    hole_new = np.array([-360, 320])
                    mu = 0.05
                    bg = Background('height_map3.png', [0, 0])
                    color_im = pygame.image.load(os.path.join(ImageFolder,'color_bar3.png'))
                    max_h = 34.84012134915605
                    switch_map = True

            if map_editor.collidepoint(mx, my):
                if click:
                    editor = True
                    show_guide = False

            if visualize.collidepoint(mx, my):
                if click:
                    if visuals == "Path":
                        visuals = 'Shot'
                        visuals_bool = True
                    elif visuals == "Shot":
                        visuals = 'Path'
                        visuals_bool = False

            if res_down.collidepoint(mx, my):
                if click:
                    if res > 2.1:
                        res -= 1
                    else:
                        res = 1

            if res_up.collidepoint(mx, my):
                if click:
                    if res < 19.1:
                        res += 1
                    else:
                        res = 20

            if bisect.collidepoint(mx, my):
                if click:
                    win = False
                    live = False
                    ready = True
                    show_guide = True
                    editor = False
                    ball.set(np.array([0, -400]), hit_ang, hit_v)
                    best_dist = ((ball.pos[0] - hole_new[0]) ** 2 + (ball.pos[1] - hole_new[1]) ** 2) ** 0.5
                    sol = bisection_search(map, ball, res, [0, 100], dt_c, (0, 100), 0)
                    if type(sol) != str:
                        hit_ang = math.radians(sol[0])
                        hit_v = sol[1]

            if brute.collidepoint(mx, my):
                if click:
                    win = False
                    live = False
                    ready = True
                    show_guide = True
                    editor = False
                    ball.set(np.array([0, -400]), hit_ang, hit_v)
                    best_dist = ((ball.pos[0] - hole_new[0]) ** 2 + (ball.pos[1] - hole_new[1]) ** 2) ** 0.5
                    minimum = math.floor(min_vel(map, [0, -400]))  - 1
                    sol = smart_brute_search(map, ball, res, minimum, dt_c, [0, 100], 1)
                    if type(sol) != str:
                        hit_ang = math.radians(sol[0])
                        hit_v = sol[1]

            if gradient.collidepoint(mx, my):
                if click:
                    win = False
                    live = False
                    ready = True
                    show_guide = True
                    editor = False
                    ball.set(np.array([0, -400]), hit_ang, hit_v)
                    best_dist = ((ball.pos[0] - hole_new[0]) ** 2 + (ball.pos[1] - hole_new[1]) ** 2) ** 0.5
                    minimum = math.floor(min_vel(map, [0, -400]))
                    sol = full_grad_descent(map, ball, minimum, res, dt_c)
                    if type(sol) != str:
                        hit_ang = math.radians(sol[0])
                        hit_v = sol[1]

            if nadam.collidepoint(mx, my):
                if click:
                    win = False
                    live = False
                    ready = True
                    show_guide = True
                    editor = False
                    ball.set(np.array([0, -400]), hit_ang, hit_v)
                    best_dist = ((ball.pos[0] - hole_new[0]) ** 2 + (ball.pos[1] - hole_new[1]) ** 2) ** 0.5
                    minimum = math.floor(min_vel(map, [0, -400]))
                    sol = nadam_full_grad_descent(map, ball, minimum, res, dt_c)
                    if type(sol) != str:
                        hit_ang = math.radians(sol[0])
                        hit_v = sol[1]

            if editor:

                if editor_exit.collidepoint(mx, my):
                    if click:
                        ball.set(np.array([0, -400]), hit_ang, hit_v)
                        best_dist = ((ball.pos[0] - hole[0]) ** 2 + (ball.pos[1] - hole[1]) ** 2) ** 0.5
                        editor = False
                        live = False
                        ready = True
                        show_guide = True
                        win = False

                if map_clear.collidepoint(mx, my):
                    if click:
                        funcs = []
                        centers = []
                        centers_draw = []
                        prev = None
                        prev2 = None
                        line = []
                        lines = []
                        lines_draw = []
                        l_prev1 = None
                        l_prev2 = None
                        map_flat = lambda x, y: 0
                        generate_map(map_flat)
                        bg = Background('height_map.png', [0, 0])
                        color_im = pygame.image.load(os.path.join(ImageFolder,'color_bar.png'))
                        map = Map(map_flat, mu, hole)

                if thickness_up.collidepoint(mx, my):
                    if click:
                        if brush_thick < 499:
                            brush_thick += 10
                        else:
                            brush_thick = 500

                if thickness_down.collidepoint(mx, my):
                    if click:
                        if brush_thick > 21:
                            brush_thick -= 10
                        else:
                            brush_thick = 10

                if height_up.collidepoint(mx, my):
                    if click:
                        if brush_height < 99:
                            brush_height += 5
                        else:
                            brush_height = 100

                if height_down.collidepoint(mx, my):
                    if click:
                        if brush_height > -99:
                            brush_height -= 5
                        else:
                            brush_height = -100

                if mode == "circle":
                    pygame.draw.circle(self.screen, (255, 0, 0, 50), (mx, my), brush_thick)
                    if click:
                        mode = 'mouse'
                        bt = brush_thick
                        bh = brush_height
                        centers.append([int(mx), int(my), int(bh), int(bt)].copy())
                        centers_draw.append([int(mx), int(my), int(bh), int(bt)].copy())
                        prev1, prev2 = [int(mx), int(my), int(bh), int(bt)].copy(), \
                                       [int(mx), int(my), int(bh), int(bt)].copy()
                        prevtype = "circle"

                if mode == "line":
                    pygame.draw.circle(self.screen, (255, 0, 0), (mx, my), brush_thick // 2)
                    bt = brush_thick
                    bh = brush_height
                    if end_point == 2:
                        pygame.draw.line(self.screen, (255, 0, 0), line[0], (mx, my), bt)
                        if click:
                            line.append([int(mx), int(my)].copy())
                            lines.append([line, int(bh), int(bt)])
                            lines_draw.append([line.copy(), int(bh), int(bt)])
                            l_prev1, l_prev2 = [line.copy(), int(bh), int(bt)], \
                                               [line.copy(), int(bh), int(bt)]
                            prevtype = 'line'
                            end_point = 0
                            line = []
                            mode = 'mouse'
                    if end_point == 1:
                        if click:
                            line = [[int(mx), int(my)].copy()]
                            end_point = 2

                if mode == "flag":
                    self.screen.blit(flag_im, [int(mx) - 8, int(my) - 24])
                    if click:
                        hole_new = np.array([int(mx) - 800, -int(my) + 450].copy())
                        hole = np.array([int(mx) - 800, -int(my) + 450].copy())
                        map = Map(map_new, mu, hole_new)
                        mode = 'mouse'

                if map_circle.collidepoint(mx, my):
                    if click:
                        mode = "circle"

                if map_line.collidepoint(mx, my):
                    if click:
                        end_point = 1
                        mode = "line"

                if flag_add.collidepoint(mx, my):
                    if click:
                        mode = "flag"

                if undo.collidepoint(mx, my):
                    if prevtype == "circle":
                        if click:
                            try:
                                prev = centers.pop(-1)
                                prev2 = centers_draw.pop(-1)
                            except IndexError:
                                prev, prev2 = None, None
                    elif prevtype == "line":
                        if click:
                            try:
                                l_prev1 = lines.pop(-1)
                                l_prev2 = lines_draw.pop(-1)
                            except IndexError:
                                l_prev1, l_prev2 = None, None

                if redo.collidepoint(mx, my):
                    if prevtype == "circle":
                        if click and prev is not None:
                            centers.append(prev)
                            centers_draw.append(prev2)
                    elif prevtype == "line":
                        if click and l_prev1 is not None:
                            lines.append(l_prev1)
                            lines_draw.append(l_prev2)

                if update_key.collidepoint(mx, my):
                    if click:
                        for center in centers:
                            if center in centers_draw:
                                funcs.append(circ(center[0], center[1], center[2], center[3]))
                        for line in lines:
                            if line in lines_draw:
                                funcs.append(line_seg(line[0][0], line[0][1], line[1], line[2]))
                        map_new = lambda x, y: sum(func(x, y) for func in funcs)
                        update_map = True
                        centers_draw = []
                        lines_draw = []

            def blit():

                pygame.draw.rect(self.screen, (128, 128, 128, 50), vision_ui)
                self.screen.blit(game_font.render(f'Guides : {vision}', True, (215, 215, 215)), (40, 17))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), vision_down)
                self.screen.blit(game_font.render('DOWN', True, (255, 0, 0)), (15, 63))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), vision_up)
                self.screen.blit(game_font.render('UP', True, (255, 0, 0)), (114, 63))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), mu_ui)
                self.screen.blit(game_font.render(f'Friction : {round(mu, 4)}', True, (215, 215, 215)), (25, 107))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), mu_down)
                self.screen.blit(game_font.render('DOWN', True, (255, 0, 0)), (15, 153))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), mu_up)
                self.screen.blit(game_font.render('UP', True, (255, 0, 0)), (114, 153))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), reset)
                self.screen.blit(game_font.render('RESET', True, (255, 0, 0)), (55, 200))

                pygame.draw.rect(self.screen, (128, 128, 128, 50), map_menu)
                self.screen.blit(game_font.render('MAP MENU', True, (215, 215, 215)), (35, 255))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), map_1)
                self.screen.blit(game_font.render('1', True, (0, 0, 255)), (27, 300))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), map_2)
                self.screen.blit(game_font.render('2', True, (0, 0, 255)), (81, 300))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), map_3)
                self.screen.blit(game_font.render('3', True, (20, 0, 255)), (135, 300))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), map_editor)
                self.screen.blit(game_font.render('Create Map', True, (0, 0, 255)), (35, 345))

                if editor:

                    for c in centers_draw:
                        pygame.draw.circle(self.screen, (0, 0, 0), (c[0], c[1]), c[3])

                    for l in lines_draw:
                        pygame.draw.line(self.screen, (0, 0, 0), l[0][0], l[0][1], l[2])

                    pygame.draw.rect(self.screen, (128, 128, 128, 50), flag_add)
                    self.screen.blit(game_font.render('FLAG', True, (255, 0, 0)), (462, 813))
                    pygame.draw.rect(self.screen, (128, 128, 128, 50), editor_exit)
                    self.screen.blit(game_font.render('EXIT', True, (255, 0, 0)), (462, 858))

                    pygame.draw.rect(self.screen, (128, 128, 128, 50), map_circle)
                    self.screen.blit(game_font.render('CIRCLE', True, (255, 0, 0)), (552, 813))
                    pygame.draw.rect(self.screen, (128, 128, 128, 50), map_line)
                    self.screen.blit(game_font.render('LINE', True, (255, 0, 0)), (564, 858))

                    pygame.draw.rect(self.screen, (128, 128, 128, 50), draw_thickness)
                    self.screen.blit(game_font.render(f'Brush Size : {brush_thick}', True, (215, 215, 215)), (646, 813))
                    pygame.draw.rect(self.screen, (128, 128, 128, 50), thickness_down)
                    self.screen.blit(game_font.render('DOWN', True, (255, 0, 0)), (648, 858))
                    pygame.draw.rect(self.screen, (128, 128, 128, 50), thickness_up)
                    self.screen.blit(game_font.render('UP', True, (255, 0, 0)), (745, 858))

                    pygame.draw.rect(self.screen, (128, 128, 128, 50), draw_height)
                    self.screen.blit(game_font.render(f'Map Height : {brush_height}', True, (215, 215, 215)),
                                     (806, 813))
                    pygame.draw.rect(self.screen, (128, 128, 128, 50), height_down)
                    self.screen.blit(game_font.render('DOWN', True, (255, 0, 0)), (808, 858))
                    pygame.draw.rect(self.screen, (128, 128, 128, 50), height_up)
                    self.screen.blit(game_font.render('UP', True, (255, 0, 0)), (907, 858))

                    pygame.draw.rect(self.screen, (128, 128, 128, 50), undo)
                    self.screen.blit(game_font.render('UNDO', True, (255, 0, 0)), (984, 813))
                    pygame.draw.rect(self.screen, (128, 128, 128, 50), redo)
                    self.screen.blit(game_font.render('REDO', True, (255, 0, 0)), (984, 858))

                    pygame.draw.rect(self.screen, (128, 128, 128, 50), map_clear)
                    self.screen.blit(game_font.render('CLEAR', True, (0, 0, 255)), (1087, 813))
                    pygame.draw.rect(self.screen, (128, 128, 128, 50), update_key)
                    self.screen.blit(game_font.render('UPDATE', True, (0, 0, 255)), (1079, 858))

                pygame.draw.rect(self.screen, (128, 128, 128, 50), solutions)
                self.screen.blit(game_font.render(f'SOLUTIONS', True, (215, 215, 215)), (30, 398))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), visualize)
                self.screen.blit(game_font.render(f'Visualize : {visuals}', True, (0, 255, 0)), (19, 444))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), angle_res)
                self.screen.blit(game_font.render(f'Angle Res : {res}', True, (215, 215, 215)), (22, 490))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), res_down)
                self.screen.blit(game_font.render('DOWN', True, (0, 255, 0)), (15, 537))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), res_up)
                self.screen.blit(game_font.render('UP', True, (0, 255, 0)), (114, 537))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), bisect)
                self.screen.blit(game_font.render('Bisection', True, (0, 255, 0)), (46, 582))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), brute)
                self.screen.blit(game_font.render('Smart Brute', True, (0, 255, 0)), (35, 628))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), gradient)
                self.screen.blit(game_font.render('Grad Descent', True, (0, 255, 0)), (25, 674))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), nadam)
                self.screen.blit(game_font.render('Nadam GD', True, (0, 255, 0)), (38, 721))

                pygame.draw.rect(self.screen, (128, 128, 128, 50), dist_bar)
                self.screen.blit(game_font.render(f'Best Dist : {round(best_dist)}', True, (215, 215, 215)), (20, 766))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), angle_bar)
                self.screen.blit(game_font.render(f'Angle : {round(math.degrees(hit_ang), 2)}', True, (215, 215, 215)),
                                 (25, 813))
                pygame.draw.rect(self.screen, (128, 128, 128, 50), vel_bar)
                self.screen.blit(game_font.render(f'Speed : {round(hit_v, 2)}', True, (215, 215, 215)), (25, 860))

                self.screen.blit(color_im, [1270, 0])
                self.screen.blit(flag_im, [hole[0] + 800 - 12, -hole[1] + 450 - 24])

            blit()

            click = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit = True
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        click = True

            pressed = pygame.key.get_pressed()

            if not live:

                if pressed[pygame.K_SPACE] and ready:
                    ball.hit(hit_ang, hit_v)
                    ready = False
                    live = True

                if show_guide:
                    test_ball = Ball(0, -400, 0, 5, 1)
                    test_ball.set(np.array([0, -400]), hit_ang, hit_v)
                    for i in range(vision):
                        if not (map.static_fric_winning(test_ball.pos) and (abs(test_ball.vel) < 0.5 + 10 * mu).all()):
                            for j in range(10):
                                test_ball.update(map, dt=dt_c)
                        rotated_ind = pygame.transform.rotate(ind_im, math.degrees(-test_ball.ang))
                        self.screen.blit(rotated_ind,
                                         [int(test_ball.pos[0]) + 800 - 8, -int(test_ball.pos[1]) + 450 - 8])

            if pressed[pygame.K_UP]:
                if hit_v < 100:
                    hit_v += 1
                else:
                    hit_v = 100

            if pressed[pygame.K_DOWN]:
                if hit_v > 1:
                    hit_v -= 1
                else:
                    hit_v = 0

            if pressed[pygame.K_RIGHT]:
                if math.degrees(hit_ang) < 90:
                    hit_ang += 1 * math.pi / 180
                else:
                    hit_ang = math.radians(90)

            if pressed[pygame.K_LEFT]:
                if math.degrees(hit_ang) > -90:
                    hit_ang -= 1 * math.pi / 180
                else:
                    hit_ang = math.radians(-90)

            ball.ang = hit_ang

            if win:
                best_dist = 0
            if live:
                ball.update(map, dt=dt_c)
                dist = ((ball.pos[0] - hole[0]) ** 2 + (ball.pos[1] - hole[1]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                if map.static_fric_winning(ball.pos) and (abs(ball.vel) < 0.5 + 10 * mu).all():
                    live = False
                    show_guide = True
                if map.win_spot(ball.pos):
                    print(f"Ang = {round(math.degrees(hit_ang))}   "
                          f"Vel = {round(hit_v)}   "
                          f"Mu = {round(mu, 4)}  "
                          f"Map = {map.name}")
                    live = False
                    show_guide = True
                    win = True

            if not editor:
                height = map.h(ball.pos[0], ball.pos[1])
                height_ind = pygame.Rect(1500, int(round(450 - height / max_h * 409)), 43, 5)
                pygame.draw.rect(self.screen, (255, 0, 0), height_ind)
                rotated = pygame.transform.rotate(ball_im, math.degrees(-ball.ang))
                self.screen.blit(rotated, [int(ball.pos[0]) + 800 - 16, -int(ball.pos[1]) + 450 - 16])

            self.clock.tick(self.ticks)
            pygame.event.pump()
            pygame.display.flip()
        pygame.quit()


def measure(map, ball, ang, v_in, dt, continuous):
    best_dist = ((ball.pos[0] - map.hole[0]) ** 2 + (ball.pos[1] - map.hole[1]) ** 2) ** 0.5
    best_pos = ball.pos
    best_vel = ball.vel
    running = True
    ball.hit(ang, v_in)
    initial_vel = ball.vel
    while running:
        ball.update(map, dt=dt)
        dist = ((ball.pos[0] - map.hole[0]) ** 2 + (ball.pos[1] - map.hole[1]) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_pos = ball.pos
            best_vel = ball.vel
        if (dist > best_dist + 300) or \
                (map.static_fric_winning(ball.pos) and (abs(ball.vel) < 0.5 + 10 * map.mu).all()):
            running = False
        if not (-800 <= ball.pos[0] <= 800 and -450 <= ball.pos[1] <= 450):
            running = False
        if not continuous:
            if map.win_spot(ball.pos):
                best_dist = 0
                ball.pos = map.hole
                running = False
        else:
            if map.win_spot(ball.pos):
                running = False
    ball.set(np.array([0, -400]), 0, 16)
    return best_dist, best_pos[0] - map.hole[0], \
           (best_vel[0] + 0.0000003141529) * initial_vel[0], (best_vel[1] + 0.0000003141529)


def cost_func(map, ball, dt):
    return lambda ang, v_in: measure(map, ball, math.radians(ang), v_in, dt, True)[0] + v_in ** 2 / 10


def cost_func_high_mu(map, ball, dt):
    return lambda ang, v_in: measure(map, ball, math.radians(ang), v_in, dt, True)[0]


def nadam_cost_func(map, ball, dt):
    return lambda ang, v_in: (measure(map, ball, math.radians(ang), v_in, dt, True)[0]) ** 2 \
                             + v_in ** (3 * (1 - map.mu / 1.5))


if __name__ == '__main__':
    # test_ball = Ball(0, -400, 0, 16, 1)
    # test_map1 = Map(map_one, 0.1, [200, 375])
    # test_map2 = Map(map_two, 0.025, [-175, 0])
    # test_map3 = Map(map_three, 0.05, [-360, 320])
    # visual_map(nadam_cost_func(test_map1, test_ball, 0.4), False)
    test = Game()
    test.run()
