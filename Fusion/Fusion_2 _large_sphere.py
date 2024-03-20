# Fusion_Grid_HemiSpheres.py
'''
This script reads the circle centers from a CSV file,
Creates spheres at each circle center in Fusion 360.

'''
import adsk.core, adsk.fusion, adsk.cam, traceback
import csv

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        ui.messageBox('Script execution has started. Please wait...')

        design = app.activeProduct
        design.designType = adsk.fusion.DesignTypes.DirectDesignType
        rootComp = design.rootComponent

        # Paths to the left and right CSV files
        left_file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\Left Side_info.csv"
        right_file_path = r"C:\Users\thoma\OneDrive - Imperial College London\Des Eng Y4\DfAM\CW2_FlipFlop\DfAM_FlipFlop\Right Side_info.csv"

        # Load circle data and create spheres
        createSphereFromCSV(rootComp, left_file_path)
        createSphereFromCSV(rootComp, right_file_path)

        ui.messageBox('Script execution completed successfully!')
    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))

def createSphereFromCSV(rootComp, file_path):
    circles_information = []
    with open(file_path, mode='r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)  # Skip the header row
        for row in csvreader:
            # circle_info columns: min_z_x,min_z_y,min_z_radius,min_z_z,max_y,min_y,greatest_distance
            circle_info = {'min_z_x': float(row[0]), 'min_z_y': float(row[1]), 'min_z_radius': float(row[2]), 'min_z_z': float(row[3]), 'max_y': float(row[4]), 'min_y': float(row[5]), 'greatest_distance': float(row[6])}
            circles_information.append(circle_info)

    # Find the circle with the maximum radius
    # big sphere top = min_z_z + min_z_radius
    info = circles_information[0]
    length = info['greatest_distance']*10
    depth = 0.5
            
    big_rad = (((length**2)/4)+(depth**2))/(2*depth)

    #centre = (info['min_z_z'] + info['min_z_radius']) - big_rad
    centre = info['min_z_z'] - big_rad

    # max_radius_circle = max(circle_centers, key=lambda x: x['radius'])
    createSphere(rootComp, info['min_z_x'], info['min_z_y'], centre, big_rad)

def createSphere(rootComp, x, y, z, radius):
    features = rootComp.features
    sketches = rootComp.sketches
    xyPlane = rootComp.xYConstructionPlane
    sketch = sketches.add(xyPlane)

    # Draw a circle
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
    angle = adsk.core.ValueInput.createByString("-180 deg")  # Revolve angle
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
