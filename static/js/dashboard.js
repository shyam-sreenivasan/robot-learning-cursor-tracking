// ============================================================
// APPLICATION STATE
// ============================================================

let experiments = [];
let experimentData = null;
let scopeData = null;
let selectedEpisode = null;

let replayFrames = [];
let replayPosition = 0;
let playbackGeneration = 0;
let navigationGeneration = 0;

const $ = id => document.getElementById(id);


// ============================================================
// UTILITIES
// ============================================================

async function api(path) {

    const response = await fetch(path);

    if (!response.ok) {
        throw new Error(
            `${response.status}: ${path}`
        );
    }

    return response.json();
}


function showError(error) {

    $("errorMessage").textContent = error.message;
    $("errorMessage").classList.remove("d-none");
}


function run(action) {

    $("errorMessage").classList.add("d-none");

    Promise.resolve()
        .then(action)
        .catch(showError);
}


function percentage(value) {

    return value == null
        ? "N/A"
        : `${(value * 100).toFixed(1)}%`;
}


function seconds(value) {

    return value == null
        ? "N/A"
        : `${Number(value).toFixed(2)} s`;
}


function shortId(value) {

    return String(value ?? "").slice(0, 8);
}


function episodePath(episode) {

    return `/api/episodes/${
        encodeURIComponent(episode.session_id)
    }/${
        encodeURIComponent(episode.episode_id)
    }`;
}


function renderMetrics(containerId, metrics) {

    const container = $(containerId);

    container.replaceChildren();

    for (const [label, value] of metrics) {

        const column = document.createElement("div");

        column.className = "col-6 col-lg-3";

        const card = document.createElement("div");
        card.className = "metric-card";

        const heading = document.createElement("div");
        heading.className = "metric-label";
        heading.textContent = label;

        const number = document.createElement("div");
        number.className = "metric-value";
        number.textContent = value ?? "N/A";

        card.append(heading, number);
        column.append(card);
        container.append(column);
    }
}


function appendCells(row, values) {

    for (const value of values) {

        const cell = document.createElement("td");

        cell.textContent = value ?? "N/A";

        row.append(cell);
    }
}


// ============================================================
// GLOBAL OVERVIEW
// ============================================================

async function initialize() {

    const generation = ++navigationGeneration;

    resetEpisode();

    const [overview, inventory] = await Promise.all([
        api("/api/overview"),
        api("/api/experiments")
    ]);

    if (generation !== navigationGeneration) return;

    experiments = inventory;

    renderMetrics("overview", [

        ["Experiments", overview.total_experiments],
        ["Configurations", overview.total_configurations],
        ["Sessions", overview.total_sessions],
        ["Episodes", overview.total_episodes],
        ["Successful", overview.successful],
        ["Failed", overview.failed],
        ["Success Rate", percentage(overview.success_rate)],
        ["Total Duration", seconds(overview.total_duration_sec)]

    ]);

    const selector = $("experimentSelect");

    selector.replaceChildren();

    experiments.forEach((experiment, index) => {

        const option = document.createElement("option");

        option.value = index;

        option.textContent = [
            experiment.experiment_id,
            experiment.name,
            shortId(experiment.config_hash),
            `${experiment.episode_count} episodes`
        ].join(" | ");

        selector.append(option);
    });

    if (experiments.length) {

        await loadExperiment(0);

    } else {

        experimentData = null;
        scopeData = null;

        $("experimentStats").replaceChildren();
        $("sessionTable").replaceChildren();
        $("sessionSelect").replaceChildren();
        $("episodeTable").replaceChildren();

        $("coverageScope").textContent =
            "No recordings available.";

        for (const id of [
            "cursorCoverage",
            "targetCoverage",
            "radiusDistribution",
            "distanceDistribution",
            "directionDistribution",
            "durationDistribution"
        ]) {

            Plotly.purge(id);
        }
    }
}


// ============================================================
// EXPERIMENT EXPLORER
// ============================================================

async function loadExperiment(index) {

    const generation = ++navigationGeneration;

    resetEpisode();

    const experiment = experiments[index];

    const result = await api(
        `/api/experiments/${
            encodeURIComponent(experiment.experiment_id)
        }/${
            encodeURIComponent(experiment.config_hash)
        }`
    );

    if (generation !== navigationGeneration) return;

    experimentData = result;
    scopeData = result;

    const stats = result.statistics;

    renderMetrics("experimentStats", [

        ["Sessions", result.total_sessions],
        ["Episodes", stats.total_episodes],
        ["Success Rate", percentage(stats.success_rate)],
        ["Average Completion",
            seconds(stats.average_completion_sec)],
        ["Fastest Completion",
            seconds(stats.fastest_episode?.duration_sec)],
        ["Slowest Completion",
            seconds(stats.slowest_episode?.duration_sec)],
        ["Successful", stats.successful],
        ["Failed", stats.failed]

    ]);

    renderSessions();
    renderScope();
}


// ============================================================
// SESSION EXPLORER
// ============================================================

function renderSessions() {

    const table = $("sessionTable");
    const selector = $("sessionSelect");

    table.replaceChildren();
    selector.replaceChildren();

    selector.add(
        new Option("All Sessions", "all")
    );

    for (const session of experimentData.sessions) {

        const stats = session.statistics;

        const row = document.createElement("tr");

        row.className = "selectable-row";

        appendCells(row, [

            shortId(session.session_id),
            session.operator_id ?? "Unknown",
            stats.total_episodes,
            stats.successful,
            stats.failed,
            percentage(stats.success_rate)

        ]);

        row.onclick = () => {

            selector.value = session.session_id;

            run(() => loadSession(session.session_id));
        };

        table.append(row);

        selector.add(
            new Option(
                shortId(session.session_id),
                session.session_id
            )
        );
    }
}


async function loadSession(sessionId) {

    const generation = ++navigationGeneration;

    resetEpisode();

    const result = sessionId === "all"
        ? experimentData
        : await api(
            `/api/sessions/${encodeURIComponent(sessionId)}`
        );

    if (generation !== navigationGeneration) return;

    scopeData = result;

    renderScope();
}


function renderScope() {

    $("coverageScope").textContent =
        scopeData.session_id
            ? `Session: ${scopeData.session_id}`
            : "All sessions in this configuration";

    renderCoverage(scopeData.coverage);

    renderDurationDistribution(scopeData.episodes);

    renderEpisodes();
}


// ============================================================
// PLOTLY VISUALIZATIONS
// ============================================================

const plotConfig = {
    responsive: true,
    displaylogo: false
};


function coverageChart(
    element,
    title,
    coverage,
    workspace,
    binsX,
    binsY
) {

    const x = Array.from(
        {length: binsX},
        (_, i) => (i + 0.5) * workspace.width / binsX
    );

    const y = Array.from(
        {length: binsY},
        (_, i) => (i + 0.5) * workspace.height / binsY
    );

    Plotly.react(element, [

        {
            type: "heatmap",
            z: coverage.grid,
            x,
            y,
            colorscale: "Blues",
            zmin: 0,
            colorbar: {
                title: "Count"
            }
        },

        {
            type: "scatter",
            mode: "markers",
            x: coverage.positions.map(p => p[0]),
            y: coverage.positions.map(p => p[1]),
            marker: {
                size: 6,
                color: "#e76f51"
            }
        }

    ], {

        title: {
            text: `${title} — ${
                percentage(coverage.coverage_ratio)
            }`
        },

        xaxis: {
            title: "X (px)",
            range: [0, workspace.width]
        },

        yaxis: {
            title: "Y (px)",
            range: [workspace.height, 0],
            scaleanchor: "x"
        },

        margin: {t: 50, l: 55, r: 25, b: 45},
        showlegend: false

    }, plotConfig);
}


function histogram(
    element,
    title,
    values,
    axisTitle
) {

    Plotly.react(element, [{

        type: "histogram",

        x: values.filter(
            value => value != null
        ),

        marker: {
            color: "#5379b5"
        }

    }], {

        title: {text: title},

        xaxis: {
            title: axisTitle
        },

        yaxis: {
            title: "Episodes"
        },

        margin: {t: 50, l: 50, r: 15, b: 50}

    }, plotConfig);
}


function renderCoverage(coverage) {

    coverageChart(
        "cursorCoverage",
        "Initial Cursor Coverage",
        coverage.cursor_coverage,
        coverage.workspace,
        coverage.bins_x,
        coverage.bins_y
    );

    coverageChart(
        "targetCoverage",
        "Target Coverage",
        coverage.target_coverage,
        coverage.workspace,
        coverage.bins_x,
        coverage.bins_y
    );

    histogram(
        "radiusDistribution",
        "Target Radius",
        coverage.target_radii_px,
        "Radius (px)"
    );

    histogram(
        "distanceDistribution",
        "Initial Distance",
        coverage.initial_distances_px,
        "Distance (px)"
    );

    histogram(
        "directionDistribution",
        "Initial Direction",
        coverage.initial_directions_deg,
        "Angle (deg)"
    );
}


function renderDurationDistribution(episodes) {

    const outcomes = [
        ["success", "#198754"],
        ["failed", "#dc3545"],
        ["aborted", "#997404"]
    ];

    const traces = outcomes.map(([status, color]) => ({

        type: "histogram",

        name: status,

        x: episodes
            .filter(e => e.status === status)
            .map(e => e.duration_sec),

        marker: {color},

        opacity: 0.75,

        xbins: {size: 0.5}

    }));

    Plotly.react("durationDistribution", traces, {

        title: {
            text: "Episode Duration by Outcome"
        },

        barmode: "overlay",

        xaxis: {
            title: "Duration (seconds)"
        },

        yaxis: {
            title: "Episodes"
        },

        margin: {t: 50, l: 55, r: 20, b: 50}

    }, plotConfig);
}


// ============================================================
// EPISODE EXPLORER
// ============================================================

function renderEpisodes() {

    const table = $("episodeTable");

    table.replaceChildren();

    if (!scopeData) return;

    const filter = $("outcomeFilter").value;

    const episodes = scopeData.episodes.filter(
        e => filter === "all" || e.status === filter
    );

    for (const episode of episodes) {

        const row = document.createElement("tr");

        row.className = "selectable-row";

        appendCells(row, [

            shortId(episode.episode_id),
            shortId(episode.session_id),
            episode.batch_number,
            episode.status,
            seconds(episode.duration_sec),
            episode.initial_distance_px.toFixed(1),
            episode.target_radius.toFixed(1)

        ]);

        row.children[3].className =
            `status-${episode.status}`;

        row.onclick = () => run(
            () => loadEpisode(episode)
        );

        table.append(row);
    }
}


function resetEpisode() {

    pauseReplay();

    selectedEpisode = null;
    replayFrames = [];
    replayPosition = 0;

    $("episodeDetails").classList.add("d-none");
    $("replayPanel").classList.add("d-none");

    $("loadReplayButton").disabled = false;

    for (const id of [
        "firstFrame",
        "lastFrame",
        "replayFrame"
    ]) {

        $(id).removeAttribute("src");
    }
}


async function loadEpisode(episode) {

    const generation = ++navigationGeneration;

    resetEpisode();

    const path = episodePath(episode);

    const [summary, trajectory] = await Promise.all([

        api(path),
        api(`${path}/trajectory`)

    ]);

    if (generation !== navigationGeneration) return;

    selectedEpisode = summary;

    $("selectedEpisodeId").textContent =
        summary.episode_id;

    renderMetrics("episodeStats", [

        ["Outcome", summary.status],
        ["Duration", seconds(summary.duration_sec)],
        ["Initial Distance",
            `${summary.initial_distance_px.toFixed(1)} px`],
        ["Target Radius",
            `${summary.target_radius.toFixed(1)} px`]

    ]);

    $("firstFrame").src =
        `${path}/frames/${summary.first_frame_id}`;

    $("lastFrame").src =
        `${path}/frames/${summary.last_frame_id}`;

    $("episodeDetails").classList.remove("d-none");

    renderTrajectory(
        trajectory,
        scopeData.coverage.workspace
    );

    $("episodeDetails").scrollIntoView({
        behavior: "smooth"
    });
}


// ============================================================
// TRAJECTORY
// ============================================================

function renderTrajectory(trajectory, workspace) {

    const [tx, ty] = trajectory.target_position;

    const radius = trajectory.target_radius;

    const last = trajectory.cursor_x.length - 1;

    Plotly.react("trajectoryChart", [

        {
            type: "scatter",
            mode: "lines+markers",
            name: "Trajectory",

            x: trajectory.cursor_x,
            y: trajectory.cursor_y,

            marker: {size: 3},

            line: {
                width: 2,
                color: "#3366cc"
            },

            customdata: trajectory.timestamps_sec,

            hovertemplate:
                "X: %{x}<br>Y: %{y}<br>" +
                "Time: %{customdata:.3f} s<extra></extra>"
        },

        {
            type: "scatter",
            mode: "markers",
            name: "Start",
            x: [trajectory.cursor_x[0]],
            y: [trajectory.cursor_y[0]],
            marker: {
                size: 11,
                color: "#198754"
            }
        },

        {
            type: "scatter",
            mode: "markers",
            name: "End",
            x: [trajectory.cursor_x[last]],
            y: [trajectory.cursor_y[last]],
            marker: {
                size: 11,
                color: "#dc3545"
            }
        }

    ], {

        title: {text: "Cursor Trajectory"},

        xaxis: {
            title: "X (px)",
            range: [0, workspace.width]
        },

        yaxis: {
            title: "Y (px)",
            range: [workspace.height, 0],
            scaleanchor: "x"
        },

        shapes: [{
            type: "circle",
            x0: tx - radius,
            x1: tx + radius,
            y0: ty - radius,
            y1: ty + radius,
            fillcolor: "rgba(25,135,84,0.15)",
            line: {color: "#198754"}
        }],

        margin: {t: 50, l: 50, r: 20, b: 50}

    }, plotConfig);
}


// ============================================================
// ON-DEMAND EPISODE REPLAY
// ============================================================

async function loadReplay() {

    if (!selectedEpisode) return;

    const episode = selectedEpisode;
    const generation = navigationGeneration;

    const frames = await api(
        `${episodePath(episode)}/frames`
    );

    if (
        generation !== navigationGeneration ||
        episode !== selectedEpisode
    ) return;

    if (!frames.length) {
        throw new Error("No replay frames available.");
    }

    replayFrames = frames;

    $("replaySlider").max = frames.length - 1;

    $("replayPanel").classList.remove("d-none");

    $("loadReplayButton").disabled = true;

    showReplayFrame(0);
}


function showReplayFrame(position) {

    if (!replayFrames.length || !selectedEpisode) {
        return;
    }

    replayPosition = Math.max(
        0,
        Math.min(position, replayFrames.length - 1)
    );

    const frame = replayFrames[replayPosition];

    const elapsed =
        frame.timestamp - replayFrames[0].timestamp;

    $("replayFrame").src =
        `${episodePath(selectedEpisode)}/frames/${frame.frame_id}`;

    $("replaySlider").value = replayPosition;

    $("replayTime").textContent =
        `${elapsed.toFixed(2)} s`;
}


function pauseReplay() {

    playbackGeneration++;
}


function playReplay() {

    if (!replayFrames.length) return;

    pauseReplay();

    const generation = playbackGeneration;

    if (replayPosition === replayFrames.length - 1) {
        showReplayFrame(0);
    }

    const speed = Number($("playbackSpeed").value);

    const startingTimestamp =
        replayFrames[replayPosition].timestamp;

    const wallStart = performance.now();

    function tick(now) {

        if (generation !== playbackGeneration) return;

        const desiredTimestamp =
            startingTimestamp +
            ((now - wallStart) / 1000) * speed;

        let position = replayPosition;

        while (
            position + 1 < replayFrames.length &&
            replayFrames[position + 1].timestamp
                <= desiredTimestamp
        ) {

            position++;
        }

        if (position !== replayPosition) {

            showReplayFrame(position);
        }

        if (position < replayFrames.length - 1) {

            requestAnimationFrame(tick);
        }
    }

    requestAnimationFrame(tick);
}


// ============================================================
// EVENT HANDLERS
// ============================================================

$("experimentSelect").onchange = event => run(
    () => loadExperiment(Number(event.target.value))
);

$("sessionSelect").onchange = event => run(
    () => loadSession(event.target.value)
);

$("outcomeFilter").onchange = renderEpisodes;

$("loadReplayButton").onclick = () => run(loadReplay);

$("playButton").onclick = playReplay;

$("pauseButton").onclick = pauseReplay;

$("restartButton").onclick = () => {

    pauseReplay();
    showReplayFrame(0);
};

$("replaySlider").oninput = event => {

    pauseReplay();

    showReplayFrame(
        Number(event.target.value)
    );
};

$("playbackSpeed").onchange = pauseReplay;

window.addEventListener(
    "DOMContentLoaded",
    () => run(initialize)
);