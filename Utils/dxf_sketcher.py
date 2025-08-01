import ezdxf  # type: ignore


def create_dxf(height, width, filename):
    success = False
    x, y = 0, 0
    corners = [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height)
    ]
    try:
        doc = ezdxf.new("R2010")
        msp = doc.modelspace()

        # Create a closed lightweight polyline rectangle
        msp.add_lwpolyline(
            corners,
            close=True,
            dxfattribs={"color": 7}
        )

        doc.saveas(filename)
        success = True
    except Exception as e:
        print(f"An error occurred: {e}")
    return success
