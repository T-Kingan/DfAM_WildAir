import adsk.core, adsk.fusion, adsk.cam, traceback
import numpy as np

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct

        # Generate a data structure of 10 x 10 x 10 of random numbers for sphere radii
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

        # Loop through each point in the meshgrid
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                for k in range(X.shape[2]):
                    # Sphere radius for the current point
                    radius = rand_radii[i, j, k]
                    # Create a sphere at the position with the random radius
                    createSphere(rootComp, X[i, j, k], Y[i, j, k], Z[i, j, k], radius)

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

def createSphere(rootComp, x, y, z, radius):
    # Access the features collection of the root component
    features = rootComp.features

    # Create a new sketch on the xy plane for the circle
    sketches = rootComp.sketches
    xyPlane = rootComp.xYConstructionPlane
    sketch = sketches.add(xyPlane)

    # Draw a circle in the sketch
    circles = sketch.sketchCurves.sketchCircles
    centerPoint = adsk.core.Point3D.create(x, y, z)
    circle = circles.addByCenterRadius(centerPoint, radius)

    # Extrude the circle to create a sphere
    prof = sketch.profiles.item(0)
    extrudes = features.extrudeFeatures
    extrudeInput = extrudes.createInput(prof, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    distance = adsk.core.ValueInput.createByReal(radius * 2)  # Extrude distance
    extrudeInput.setDistanceExtent(False, distance)
    extrudeInput.isSolid = True
    extrudes.add(extrudeInput)

if __name__ == "__main__":
    run(adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct))
    