import warnings
import timeit
import pandas as pd
from numpy import (
    linspace,
    array,
    where,
    mean,
    sum,
    diff,
    any,
    cumsum,
    percentile,
)
from numpy.random import normal
from scipy.stats import gaussian_kde
import plotly.graph_objects as go


INTUIT_RED = "#bd0707"
INTUIT_BLUE = "#0177c9"

def draw_distribution_classic(
    posterior_samples=None,
    title=None,
    is_rate=0,
    colors=None,
    add_median=False,
    x_axis_title=None,
    positive_direction=True,
    is_image=False,
    run_sci=True,
):
    histogram_samples = (
        normal(0, 1, 10000) if posterior_samples is None else posterior_samples
    )
    MARKER_SIZE_IMAGE = 100

    final_colors = [INTUIT_RED, INTUIT_BLUE] if colors is None else colors

    ll = min(histogram_samples)
    ul = max(histogram_samples)

    number_points = 125 if is_image else 250

    start = timeit.default_timer()

    if run_sci:
        pdf = gaussian_kde(histogram_samples)
        X = linspace(min(histogram_samples), max(histogram_samples), number_points)
        estimated_density = pdf.evaluate(X)
        cumulative_density = cumsum(estimated_density)
        cumulative_density /= cumulative_density[-1]
        cdf_list = list(cumulative_density)
        fx_list = estimated_density

    else:
        X = list(linspace(ll, ul, number_points))

        cdf_list = []

        for x in X:
            cdf = sum(where(histogram_samples <= x, 1, 0))
            cdf_list.append(cdf)
        cdf_list = array(cdf_list) / len(histogram_samples)
        fx_list = list(diff(cdf_list))
        fx_list.append(0)

    stop = timeit.default_timer()

    # print(f"RUN TIME {stop-start} secs")

    plotPanda = pd.DataFrame(
        list(zip(X, fx_list, cdf_list)), columns=["x", "fX", "cdf"]
    )

    plotPanda["my_color"] = where(plotPanda["x"] < 0, "red", "blue")
    cum_dist_string = "P( θ > x)"
    plotPanda[cum_dist_string] = 1 - array(cdf_list)
    plotPanda[cum_dist_string] = [
        "{0:.2%}".format(x) for x in plotPanda[cum_dist_string]
    ]
    positive_threshold = 100
    hover_string = (
        "True Effect"
        if (any(array(X) < positive_threshold) and any(array(X) >= positive_threshold))
        else "True Baseline"
    )
    if positive_direction:
        text_list = [
            f"Pr( {hover_string} > { round((100 if is_rate else 1)*X[j],3 if is_rate else 3) }{'%' if is_rate else ''}) = {'{0:.2%}'.format(1 - cdf_list[j])}"
            for j in range(len(X))
        ]
    else:
        text_list = [
            f"Pr( {hover_string} < { round((100 if is_rate else 1)*X[j],3 if is_rate else 3) }{'%' if is_rate else ''}) = {'{0:.2%}'.format(cdf_list[j])}"
            for j in range(len(X))
        ]
    if len(where(array(X) > positive_threshold)[0]) > 0:
        positive_position = where(array(X) > positive_threshold)[0][0]
    else:
        positive_position = len(X) - 1
    distribution_figure = go.Figure(
        [
            go.Scatter(
                x=X[:positive_position],
                y=fx_list[:positive_position],
                hoverinfo="skip",
                # marker_symbol="square",
                name=f"< {positive_threshold}",
                marker=dict(size=50),
                showlegend=False,
                line_color=colors[0] if colors is not None else "gray",
                hovertext=text_list[:positive_position],
                # fill = "blue",
                line=dict(width=0.05),
                stackgroup="1",  # define stack group
            ),
            go.Scatter(
                x=X[positive_position:],
                y=fx_list[positive_position:],
                hoverinfo="none",
                marker=dict(
                    size=44 if is_image else 18,
                    # line=dict(color="black", width=10 if is_image else 3),
                ),
                name=f"> {positive_threshold}",
                showlegend=False,
                line_color=(colors[len(colors) - 1] if colors is not None else "gray"),
                hovertext=text_list[positive_position:],
                line=dict(width=0.05),
                stackgroup="1",  # define stack group
            ),
            go.Scatter(
                x=X,
                y=fx_list,
                hoverinfo="text",
                # fill='tozeroy',
                mode="lines",
                # name = "+" ,
                showlegend=False,
                line_color="black",
                hovertext=text_list,
                # fill = "blue",
                line=dict(width=1.5),
                # stackgroup="2",  # define stack group
            ),
            go.Scatter(
                x=[None],
                y=[None],
                name=f"< {positive_threshold}",
                showlegend=(
                    True
                    if (
                        any(array(X) < positive_threshold)
                        and any(array(X) >= positive_threshold)
                    )
                    else False
                ),
                line=dict(width=24, color="#e28e8b" if colors is not None else "gray"),
                legendgroup="Thick Lines",
                mode="markers",
                marker_symbol="square",
                marker=dict(
                    size=MARKER_SIZE_IMAGE if is_image else 18,
                ),
            ),
            go.Scatter(
                x=[None],
                y=[None],
                showlegend=(
                    True
                    if (
                        any(array(X) < positive_threshold)
                        and any(array(X) >= positive_threshold)
                    )
                    else False
                ),
                name=f"> {positive_threshold}",
                line=dict(width=24, color="#8fbbe4" if colors is not None else "gray"),
                mode="markers",
                marker_symbol="square",
                marker=dict(
                    size=MARKER_SIZE_IMAGE if is_image else 18,
                ),
            ),
        ]
    )

    if add_median:
        distribution_figure.add_trace(
            go.Scatter(
                hoverinfo="skip",
                # name="μ",
                name="avg",
                x=[mean(histogram_samples)],
                y=[0.000],
                mode="markers",
                marker_symbol="circle-dot",
                showlegend=False,
                marker=dict(
                    color="green",
                    size=50 if is_image else 18,
                    line=dict(color="black", width=10 if is_image else 3),
                ),
            )
        )
        distribution_figure.add_trace(
            go.Scatter(
                hoverinfo="skip",
                # name="μ",
                name="avg",
                x=[None],
                y=[None],
                mode="markers",
                marker_symbol="circle-dot",
                marker=dict(
                    color="green",
                    size=MARKER_SIZE_IMAGE if is_image else 18,
                    line=dict(color="black", width=10 if is_image else 3),
                ),
            )
        )
    distribution_figure.update_layout(
        autosize=True,
        margin=dict(l=0, r=0, b=100 if is_image else 35, t=35),
        hovermode="x",
        xaxis=dict(tickfont=dict(size=37 if is_image == 1 else 13)),
        title="" if title is None else title,
        legend=dict(
            font=dict(size=65 if is_image else 15),
            x=0.02,
        ),
    )
    if is_rate:
        if not is_image:
            distribution_figure.update_xaxes(tickformat=".2%")
        else:
            distribution_figure.update_xaxes(tickformat=".2%")
    else:
        # if (any(array(X) < 0) and any(array(X) >= 0) and (is_image)):
        if not is_image:
            distribution_figure.update_xaxes(tickformat=".2f")
        else:
            distribution_figure.update_xaxes(tickformat=".2f")
    tick_values = [
        percentile(histogram_samples, 2.5),
        percentile(histogram_samples, 50),
        percentile(histogram_samples, 97.5),
    ]
    distribution_figure.update_xaxes(
        zeroline=False,
        tickangle=0,
        title="" if x_axis_title is None else x_axis_title,
        tickvals=tick_values,
        title_font=dict(size=35 if is_image else 15),
        ticks="inside",  # Default is 'outside'
        ticklen=30 if is_image else 6,
    )
    distribution_figure.update_yaxes(
        title="",
        visible=False,
        showticklabels=False,
    )
    distribution_figure.update_traces(hoverlabel=dict(font=dict(size=19)))
    return distribution_figure

