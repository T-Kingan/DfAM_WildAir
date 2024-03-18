import numpy as np
import adsk.core, adsk.fusion, adsk.cam, traceback

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct

        # Generate a data structure of 10 x 10 x 10 of random numbers for sphere radii
        rand_radii = np.random.rand(10, 10, 10)

        # Define the spacing between points
        spacing = 8.382

        # Create a coordinate grid for the data
        x = np.arange(0, 10 * spacing, spacing)
        y = np.arange(0, 10 * spacing, spacing)
        z = np.arange(0, 10 * spacing, spacing)
        X, Y, Z = np.meshgrid(x, y, z)

        # Access the root component of the design
        rootComp = design.rootComponent
        # Get the 3D sketch features
        sketches = rootComp.sketches
        # Create a new sketch on the xy plane
        xyPlane = rootComp.xYConstructionPlane
        sketch = sketches.add(xyPlane)

        # Loop through each point in the meshgrid
        for i in range(X.shape[0]):
            for j in range(Y.shape[1]):
                for k in range(Z.shape[2]):
                    # Sphere radius for the current point, scaled for visibility
                    radius = rand_radii[i, j, k] * 10  # Scale factor for radius
                    
                    # Create a sphere at the position with the random radius
                    createSphere(rootComp, X[i, j, k], Y[i, j, k], Z[i, j, k], radius)

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

def createSphere(rootComp, x, y, z, radius):
    # Create a temporary occurrence for a new component
    tempOcc = rootComp.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    tempComp = tempOcc.component

    # Create a sphere
    spheres = tempComp.features.sphereFeatures
    sphereInput = spheres.createInput(tempComp.xYConstructionPlane, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    sphereInput.setByCenterRadius(adsk.core.Point3D.create(x, y, z), radius)
    sphere = spheres.add(sphereInput)

    # Optionally, delete the temp occurrence if not needed
    # rootComp.occurrences.item(rootComp.occurrences.count - 1).deleteMe()

if __name__ == "__main__":
    run(adsk.fusion.Design.cast(adsk.core.Application.get().activeProduct))
