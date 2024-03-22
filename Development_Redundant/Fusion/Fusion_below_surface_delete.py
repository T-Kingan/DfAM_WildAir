import adsk.core, adsk.fusion, adsk.cam, traceback

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface
        design = app.activeProduct

        # Check if the design is not in direct modeling mode
        if design.designType == adsk.fusion.DesignTypes.ParametricDesignType:
            ui.messageBox('This script is only compatible with direct modeling mode.')
            return

        selection = ui.activeSelections
        bodies = [ent for ent in selection if isinstance(ent, adsk.fusion.BRepBody)]
        surface = next((ent for ent in selection if isinstance(ent, adsk.fusion.BRepFace)), None)

        if not surface:
            ui.messageBox("Please select at least one surface.")
            return

        # Get the reference surface Z position
        surface_point = surface.pointOnFace
        surface_z = surface_point.z

        # Start a transaction
        with adsk.fusion.Design.cast(app.activeProduct).timeline.beginTransactionGroup() as tg:
            for body in bodies:
                # Check each body's bounding box min point against the surface Z position
                min_point = body.boundingBox.minPoint
                if min_point.z < surface_z:
                    # Delete body if it's below the surface
                    body.deleteMe()

            # End the transaction
            tg.end()
            ui.messageBox("Completed. Bodies below the selected surface have been deleted.")

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
