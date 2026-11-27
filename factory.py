from io import BytesIO

from flask import Flask, jsonify, request, send_file
from PIL import Image

from src.core.config.config_manager import ConfigManager
from src.data_factory.loaders.data_catalog import DataCatalog
from src.data_factory.loaders.episode_loader import EpisodeLoader

from src.data_factory.analytics.analytics import (
    get_overview,
    get_experiment_stats,
    get_session_stats,
    get_episode_summary,
    get_episode_trajectory
)

from flask import render_template

app = Flask(__name__)

config = ConfigManager("configs/experiment_001.yaml")

catalog = DataCatalog(
    config.resolve_path(
        config.get("storage.raw_data_dir")
    )
)


@app.get("/")
def dashboard():
    return render_template("dashboard.html")

# ============================================================
# GLOBAL INVENTORY
# ============================================================

@app.get("/api/overview")
def overview():

    return jsonify(get_overview(catalog))


@app.get("/api/experiments")
def experiments():

    return jsonify(catalog.list_experiments())


@app.get("/api/sessions")
def sessions():

    return jsonify(
        catalog.list_sessions(
            experiment_id=request.args.get("experiment_id"),
            config_hash=request.args.get("config_hash")
        )
    )


# ============================================================
# EXPERIMENT ANALYTICS
# ============================================================

@app.get("/api/experiments/<experiment_id>/<config_hash>")
def experiment_stats(experiment_id, config_hash):

    return jsonify(
        get_experiment_stats(
            catalog,
            experiment_id,
            config_hash
        )
    )


# ============================================================
# SESSION ANALYTICS
# ============================================================

@app.get("/api/sessions/<session_id>")
def session_stats(session_id):

    return jsonify(
        get_session_stats(catalog, session_id)
    )


@app.get("/api/sessions/<session_id>/episodes")
def session_episodes(session_id):

    return jsonify(
        catalog.list_episodes(session_id)
    )


# ============================================================
# EPISODE ANALYTICS
# ============================================================

@app.get("/api/episodes/<session_id>/<episode_id>")
def episode_summary(session_id, episode_id):

    loader = EpisodeLoader(
        catalog, session_id, episode_id
    )

    return jsonify(
        get_episode_summary(loader)
    )


@app.get("/api/episodes/<session_id>/<episode_id>/trajectory")
def episode_trajectory(session_id, episode_id):

    loader = EpisodeLoader(
        catalog, session_id, episode_id
    )

    return jsonify(
        get_episode_trajectory(loader)
    )


# ============================================================
# OBSERVATION ACCESS
# ============================================================

@app.get("/api/episodes/<session_id>/<episode_id>/frames")
def observation_index(session_id, episode_id):

    loader = EpisodeLoader(
        catalog, session_id, episode_id
    )

    index = loader.load_observation_index()

    return jsonify(
        index[["frame_id", "timestamp"]]
        .to_dict(orient="records")
    )


@app.get(
    "/api/episodes/<session_id>/<episode_id>/frames/<int:frame_id>"
)
def observation(session_id, episode_id, frame_id):

    loader = EpisodeLoader(
        catalog, session_id, episode_id
    )

    image = loader.load_observation(frame_id)

    buffer = BytesIO()

    Image.fromarray(image).save(
        buffer,
        format="PNG"
    )

    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="image/png"
    )


# ============================================================
# ERROR HANDLING
# ============================================================

@app.errorhandler(KeyError)
@app.errorhandler(FileNotFoundError)
def handle_not_found(error):

    return jsonify({
        "error": "Requested data not found"
    }), 404


@app.errorhandler(ValueError)
def handle_invalid_request(error):

    return jsonify({
        "error": str(error)
    }), 400


# ============================================================
# LEARNING ARTIFACTS (read only)
# ============================================================
from src.data_factory.loaders.learning_catalog import LearningCatalog

learning = LearningCatalog(config.resolve_path(config.get("storage.processed_data_dir")))

@app.get("/learning")
def learning_dashboard():
    return render_template("learning.html")

@app.get("/api/learning")
def learning_inventory():
    return jsonify(learning.inventory())

@app.get("/api/learning/<dataset>")
def learning_dataset(dataset):
    return jsonify(learning.dataset(dataset))

@app.get("/api/learning/<dataset>/<experiment>")
def learning_experiment(dataset, experiment):
    return jsonify(learning.experiment(dataset, experiment))

@app.get("/api/learning/<dataset>/<experiment>/<run>")
def learning_run(dataset, experiment, run):
    return jsonify(learning.run(dataset, experiment, run))

@app.get("/api/learning/<dataset>/<experiment>/<run>/<evaluation>")
def learning_evaluation(dataset, experiment, run, evaluation):
    return jsonify(learning.evaluation(dataset, experiment, run, evaluation))

@app.get("/api/learning/<dataset>/<experiment>/<run>/<evaluation>/episodes/<int:number>")
def learning_trajectory(dataset, experiment, run, evaluation, number):
    return jsonify(learning.trajectory(dataset, experiment, run, evaluation, number))


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )