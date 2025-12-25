from manim import *


class SquareToCircleToTriangle(Scene):
    def construct(self):
        circle = Circle()  # create a circle
        circle.set_fill(PINK, opacity=0.5)  # set color and transparency

        square = Square()  # create a square
        square.set_fill(BLUE, opacity=0.5)  # set color and transparency
        square.flip(RIGHT)  # flip horizontally
        square.rotate(-3 * TAU / 8)  # rotate a certain amount

        triangle = Triangle()
        triangle.set_fill(GREEN, opacity=0.5)  # set color and transparency
        triangle.rotate(PI / 2)  # rotate a certain amount

        self.play(Create(square))  # animate the creation of the square
        self.play(Transform(square, circle))  # interpolate the square into the circle
        self.play(Transform(square, triangle))  # interpolate the circle into the triangle
        self.play(FadeOut(square))  # fade out animation
