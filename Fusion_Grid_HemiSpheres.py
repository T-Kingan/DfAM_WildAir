import adsk.core, adsk.fusion, adsk.cam, traceback
import numpy as np
from circles_from_pressureMap import calculate_circle_centers, load_and_process_data


def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        # At the beginning of the `run` function to indicate the script is starting
        ui.messageBox('Script execution has started. Please wait...')

        design = app.activeProduct


        # Generate a data structure of random numbers for hemisphere radii
        rand_radii = np.random.rand(2, 2, 2) * 2  # Example random radii, scaled

        # Define the spacing between points
        spacing = 8.382

        # Create a coordinate grid for the data
        x = np.arange(0, 2 * spacing, spacing)
        y = np.arange(0, 2 * spacing, spacing)
        z = np.arange(0, 2 * spacing, spacing)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Access the root component of the design
        rootComp = design.rootComponent

        file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\TK_footpressure.csv"
        foot_pressure = load_and_process_data(file_path)
        circle_centers = calculate_circle_centers(foot_pressure)

        print(circle_centers)

        # # Create a hemisphere at each point in circle_centers
        for circle in circle_centers:
            createHemisphere(rootComp, circle['x'], circle['y'], circle['z'], circle['radius'])




        # # Loop through each point in the meshgrid
        # for i in range(X.shape[0]):
        #     for j in range(X.shape[1]):
        #         for k in range(X.shape[2]):
        #             # Hemisphere radius for the current point
        #             radius = rand_radii[i, j, k]
        #             # Create a hemisphere at the position with the random radius
        #             createHemisphere(rootComp, X[i, j, k], Y[i, j, k], Z[i, j, k], radius)

        # At the end of the `run` function to indicate completion
        ui.messageBox('Script execution completed successfully!')
        
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

def createHemisphere(rootComp, x, y, z, radius):
    # Access the features collection of the root component
    features = rootComp.features

    # Create a new sketch on the xy plane
    sketches = rootComp.sketches
    xyPlane = rootComp.xYConstructionPlane
    sketch = sketches.add(xyPlane)

    # Draw a circle in the sketch
    circles = sketch.sketchCurves.sketchCircles
    centerPoint = adsk.core.Point3D.create(x, y, z)
    circle = circles.addByCenterRadius(centerPoint, radius)

    # Draw a center line for the revolution
    lines = sketch.sketchCurves.sketchLines
    axis = lines.addByTwoPoints(adsk.core.Point3D.create(x, y - radius, z), adsk.core.Point3D.create(x, y + radius, z))

    # Revolve the half circle to create a hemisphere
    prof = sketch.profiles.item(0)
    revolves = features.revolveFeatures
    revolveInput = revolves.createInput(prof, axis, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    angle = adsk.core.ValueInput.createByString("180 deg")  # Revolve angle
    revolveInput.setAngleExtent(False, angle)
    revolveInput.isSolid = True
    revolves.add(revolveInput)

def stop(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        ui.messageBox('Stop addin')
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

# if __name__ == "__main__":
#     #ui.messageBox('Script execution has started from main. Please wait...')
#     run(adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct))

# CAD Steps
    # sketch circle
    # draw centre line
    # revolve half the circle 180 degrees
