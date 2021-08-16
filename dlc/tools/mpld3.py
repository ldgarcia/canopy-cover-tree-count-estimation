import mpld3


__all__ = ["ZoomSizePlugin"]


class ZoomSizePlugin(mpld3.plugins.PluginBase):
    """Adapted from: https://stackoverflow.com/a/34800275"""

    JAVASCRIPT = r"""
    // little save icon
    var my_icon = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3gUTDC0v7E0+LAAAAB1pVFh0Q29tbWVudAAAAAAAQ3JlYXRlZCB3aXRoIEdJTVBkLmUHAAAAa0lEQVQoz32QQRLAIAwCA///Mz3Y6cSG4EkjoAsk1VgAqspecVP3TTIA6MHTQ6sOHm7Zm4dWHcC4wc3hmVzT7xEbYf66dX/xnEOI7M9KYgie6qvW6ZH0grYOmQGOxzCEQn8C5k5mHAOrbeIBWLlaA3heUtcAAAAASUVORK5CYII=";

    // create plugin
    mpld3.register_plugin("zoomSize", ZoomSizePlugin);
    ZoomSizePlugin.prototype = Object.create(mpld3.Plugin.prototype);
    ZoomSizePlugin.prototype.constructor = ZoomSizePlugin;
    ZoomSizePlugin.prototype.requiredProps = [];
    ZoomSizePlugin.prototype.defaultProps = {}

    function ZoomSizePlugin(fig, props){
        mpld3.Plugin.call(this, fig, props);

        // create save button
        var SaveButton = mpld3.ButtonFactory({
            buttonID: "save",
            sticky: false,
            onActivate: function(){save_zoom(fig);}.bind(this),
            icon: function(){return my_icon;},
        });
        this.fig.buttons.push(SaveButton);
    };

    function save_zoom(fig) {
      for (const ax of fig.axes) {
        var left = ax.x.invert(ax.lastTransform.invertX(0));
        var right = ax.x.invert(ax.lastTransform.invertX(ax.width));
        var bottom = ax.y.invert(ax.lastTransform.invertY(ax.height));
        var top = ax.y.invert(ax.lastTransform.invertY(0));
        var extent = left + ", " + right + ", " + bottom + ", " + top;
        prompt("Extent of zoomed axis " + ax.axnum, extent);
      }
    }
    """

    def __init__(self):
        self.dict_ = {"type": "zoomSize"}
