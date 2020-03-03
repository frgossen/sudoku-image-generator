#!/usr/bin/env python3


import argparse
import collections
import copy
import json
import os
import random

import cv2
import numpy as np


class Sudoku:

    def __init__(self):
        self.field = collections.defaultdict(lambda: set(range(9)))

    def get_values(self, i, j):
        return self.field[(i, j)]

    def get_unique_value(self, i, j):
        assert self.is_unique_value(i, j)
        return next(iter(self.get_values(i, j)))

    def is_unique_value(self, i, j):
        return len(self.field[(i, j)]) == 1

    def is_any_value(self, i, j):
        return len(self.field[(i, j)]) == 9

    def is_no_value(self, i, j):
        return len(self.field[(i, j)]) == 0

    def set_values(self, i, j, values):
        self.field[(i, j)] = values

    def set_unique_value(self, i, j, val):
        self.field[(i, j)] = {val}

    def set_any_value(self, i, j):
        self.field[(i, j)] = set(range(9))

    def set_no_value(self, i, j):
        self.field[(i, j)] = set()

    def copy(self):
        cpy = Sudoku()
        cpy.field = copy.deepcopy(self.field)
        return cpy

    def __str__(self):
        str_repr = ""
        for i in range(9):
            for j in range(9):
                if self.is_no_value(i, j):
                    str_repr += "!"
                elif self.is_unique_value(i, j):
                    str_repr += str(self.get_unique_value(i, j))
                else:
                    str_repr += "."
            str_repr += os.linesep
        return str_repr


class SimpleSolver:

    def __init__(self, sudoku):
        self.sudoku = sudoku
        self.queue = None

    def run(self):
        self.queue = [(i, j) for i in range(9) for j in range(9) if
                      self.sudoku.is_unique_value(i, j) or self.sudoku.is_no_value(i, j)]

        while len(self.queue) > 0:
            i, j = self.queue.pop(0)

            val = None
            if self.sudoku.is_unique_value(i, j):
                val = self.sudoku.get_unique_value(i, j)

            i0 = i - i % 3
            j0 = j - j % 3

            for k in range(9):

                if k != i:
                    self.update_arc(i, j, k, j, val)

                if k != j:
                    self.update_arc(i, j, i, k, val)

                iii = i0 + k % 3
                jjj = j0 + k // 3
                if i != iii and j != jjj:
                    self.update_arc(i, j, iii, jjj, val)

        return self

    def update_arc(self, k, l, i, j, val):

        # if self.sudoku.is_unique_value(i, j):
        #     print (val, self.sudoku.get_unique_value(i, j))
        #     assert val == self.sudoku.get_unique_value(i, j)

        prev_len = len(self.sudoku.get_values(i, j))

        if val is not None:

            cell = self.sudoku.get_values(i, j)
            if val in cell:
                cell.remove(val)

        else:
            self.sudoku.set_values(i, j, set())

        if len(self.sudoku.get_values(i, j)) < prev_len and len(self.sudoku.get_values(i, j)) <= 1:
            self.queue.append((i, j))


class Creator:

    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def create_random_sudoku(self):
        sudoku = self.create_random_sudoku_solution()
        self.derive_random_deterministic_sudoku_task(sudoku)
        return sudoku

    def create_random_sudoku_solution(self):
        sudoku = Sudoku()
        while True:

            # For the generation of a sudoku solution, we will always use the simple solver.
            SimpleSolver(sudoku).run()

            # Once the sudoku is solved we have found a random sudoku solution.
            if self.is_solved(sudoku):
                return sudoku

            # When the sudoku becomes inconsistent, we start over.
            if not self.is_consistent(sudoku):
                sudoku = Sudoku()

            # Populate random field.
            underdefined_cells = [(i, j) for i in range(9) for j in range(9) if not sudoku.is_unique_value(i, j)]
            assert len(underdefined_cells) > 0
            i, j = self.random.choice(underdefined_cells)
            val = self.random.choice(list(sudoku.get_values(i, j)))
            sudoku.set_unique_value(i, j, val)

    def derive_random_deterministic_sudoku_task(self, sudoku):

        # Iterate over cells in random order.
        cells = [(i, j) for i in range(9) for j in range(9)]
        self.random.shuffle(cells)

        for i, j in cells:
            assert sudoku.is_unique_value(i, j)

            # Check if unique cell value is needed for a deterministic sudoku task.
            sudoku_tmp = sudoku.copy()
            sudoku_tmp.set_any_value(i, j)
            SimpleSolver(sudoku_tmp).run()
            if self.is_solved(sudoku_tmp):
                sudoku.set_any_value(i, j)

    @staticmethod
    def is_solved(sudoku):
        for i in range(9):
            for j in range(9):
                if not sudoku.is_unique_value(i, j):
                    return False
        return True

    @staticmethod
    def is_consistent(sudoku):
        for i in range(9):
            for j in range(9):
                if sudoku.is_no_value(i, j):
                    return False
        return True


class Template:

    def __init__(self, path):
        self.path = path
        self.background = None
        self.digits = None
        self.quadrangle = None

    def load(self):
        # Load background image.
        background_path = os.path.join(self.path, "background.png")
        self.background = cv2.imread(background_path, cv2.IMREAD_UNCHANGED).astype(np.float) / 255

        # Load the 9 digit images.
        self.digits = []
        for i in range(9):
            digit_path = os.path.join(self.path, str(i) + ".png")
            digit_mat = cv2.imread(digit_path, cv2.IMREAD_UNCHANGED).astype(np.float) / 255
            self.digits.append(digit_mat)

        # Load quadrangle.json coordinates.
        quadrangle_path = os.path.join(self.path, "quadrangle.json")
        with open(quadrangle_path, "r") as fin:
            self.quadrangle = json.load(fin)

        return self


class Embedding:

    def __init__(self, template, sudoku):
        self.template = template
        self.sudoku = sudoku
        self.canvas = None

    def render(self, highlight_quadrangle=False):

        # Determine dimensions for the intermediate rectangular rendering.
        xs, ys = [], []
        for x, y in self.template.quadrangle:
            xs.append(x)
            ys.append(y)
        intermediate_width, intermediate_height = max(xs) - min(xs), max(ys) - min(ys)
        intermediate_width *= 9
        intermediate_height *= 9

        # Determine perspective transformation.
        points_rectangular = np.float32(
            [[0, 0], [intermediate_width, 0], [intermediate_width, intermediate_height], [0, intermediate_height]])
        points = np.float32(self.template.quadrangle)
        transformation = cv2.getPerspectiveTransform(points_rectangular, points)

        # Transform rendering.
        width, height, cn = self.template.background.shape
        assert cn == 4
        intermediate = self.render_rectangularly(intermediate_width, intermediate_height)
        intermediate_transformed = cv2.warpPerspective(intermediate, transformation, (height, width))

        # Compose rendering.
        self.canvas = self.compose(intermediate_transformed, self.template.background)

        # Highlight quadrangle.json.
        if highlight_quadrangle:
            opaque_red = (0, 0, 1, 1)
            thickness = 1
            cv2.polylines(self.canvas, np.int32([self.template.quadrangle]), thickness, opaque_red)

        return self

    def render_rectangularly(self, width, height):

        # Determine cell size.
        assert width % 9 == 0 and height % 9 == 0
        cell_width, cell_height = width // 9, height // 9

        # Resize digits to cell size.
        resized_digits = []
        for mat in self.template.digits:
            resized_mat = cv2.resize(mat, (cell_width, cell_height), cv2.INTER_NEAREST)
            resized_digits.append(resized_mat)

        # Compose sudoku.
        canvas = np.zeros((height, width, 4), dtype=np.float)
        for i in range(9):
            for j in range(9):
                if self.sudoku.is_unique_value(i, j):
                    val = self.sudoku.get_unique_value(i, j)
                    x, y = j * cell_width, i * cell_height
                    canvas[y:y + cell_height, x:x + cell_width] = resized_digits[val]

        return canvas

    @staticmethod
    def compose(a, b):

        # Assert same shape.
        assert a.shape == b.shape
        shape = a.shape

        # Extract alpha channels.
        a_alpha = a[:, :, 3]
        b_alpha = b[:, :, 3]

        # Compose alpha channels.
        result_alpha = 1 - (1 - a_alpha) * (1 - b_alpha)

        # Compose color channels.
        result = np.zeros(shape, np.float)
        for i in range(3):
            np.divide((a_alpha * a[:, :, i] + (1 - a_alpha) * b_alpha * b[:, :, i]), result_alpha, result[:, :, i],
                      where=result_alpha != 0)

        # Merge result color and alpha channel.
        result[:, :, 3] = result_alpha

        return result

    def save(self, path):
        cv2.imwrite(path, 255 * self.canvas)
        return self


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="Generate Sudoku puzzles that mimics the look and feel of your daily newspaper.")
    parser.add_argument("-t", "--template", metavar="PATH", type=str, required=True,
                        help="path to the template directory")
    parser.add_argument("-s", "--seed", metavar="INT", type=int, help="random integer seed")
    parser.add_argument("-o", "--out-file", metavar="PATH", type=str, required=True, help="path to output file")
    parser.add_argument("-x", "--highlight-quadrangle", action="store_true", help="highlight quadrangle")
    args = parser.parse_args()

    # Load template.
    chosen_template = Template(args.template).load()

    # Create sudoku puzzle.
    creator = Creator(seed=args.seed)
    generated_sudoku = creator.create_random_sudoku()

    # Render sudoku puzzle.
    embedding = Embedding(chosen_template, generated_sudoku).render(
        highlight_quadrangle=args.highlight_quadrangle).save(args.out_file)

    cv2.imshow("hey", embedding.canvas)
    cv2.waitKey(0)
