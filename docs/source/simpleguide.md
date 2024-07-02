# A Simple Guide

## Installation

1. Download `zip` file from Github

2. Unzip files at any location, run `python setup.py install` at the folder where `setup.py` exists

> if you want to modify the code for development, please use `python setup.py develop`

## Run MOC code

An example python file to run MOC is provided in `/examples/moc`. The problem is described as: given the boundaries (upper and lower) and inlet, solve for the flowfield between the upper and lower boundaries.

1. Describe the boundary

    The boundary of the computation domain are described with a series of points in the class `BoundPoints`.  It is constructed by using `add_section`. The code below constructs a boundary with two sections. The first section is an 20 degree arc with radius = 6.0 and center located at (0., 8.). The second section is a straight line whose x projection length = 5. and along the  tangential direction of the upstream arc. 

    ```python
    kttau = 20.0
    upperwall = BoundPoints()
    upperwall.add_section(6 * np.sin(np.linspace(0., math.pi / 180. * kttau, 15)), lambda x: 8. - (6.**2 - x**2)**0.5)
    upperwall.add_section(np.linspace(0, 5, 16), lambda x: math.tan(math.pi / 180. * kttau) * x)
    ```

    > The boundary points are also the initial points for the character lines, so they are expected to be evenly distributed along the boundaries rather than in x-direction.

    For details in boundary conditions, please see [Boundary Conditions](./tutorials.md#boundary-conditions).

2. Define the problem

    The problem is defined with the object `MOC2D`. Here, we set up a computional domain where the upper boundary is the wall described in advance, and the lower boundary is a symmetric one.

    ```python
    moc = MOC2D()
    moc.set_boundary('u', typ='wall', points=upperwall)
    moc.set_boundary('l', typ='sym',  y0=0.0)

    ```

3. Define the initial line (inlet)

    The MOC calculation requires a initial line (in the present program, the left side of computation domain). Here, we use the method in NASA's code to describe the inlet given the inlet total condition and upstream wall radius. 

    ```python
    moc.calc_initial_line(n, mode='total', p=2015., t=2726., urUp=9., lrUp=3.)
    ```

    For details in boundary conditions, please see [Initial Methods](./tutorials.md#initial-methods).

4. Solve for field

    The problem is solved with:

    ```python
    moc.solve(max_step=100)
    ```

5. Show and save the results

    ```
    moc.plot_field().show()
    moc.plot_wall(side='u', var=['p', 'ma', 't'], write_to_file='upper.dat')
    ```

    For details in boundary conditions, please see [Post-process](./tutorials.md#post-process).


