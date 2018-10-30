# noinspection PyProtectedMember
from flask import current_app, _app_ctx_stack
import tensorflow as tf
import cloudpickle
import os


def _get_last_file(path):
    if path is None:
        raise ValueError()
    last_name = sorted(os.listdir(path))[-1]
    return os.path.join(path, last_name)


# noinspection PyMethodMayBeStatic
class SavedModel:
    _SAVED_MODEL_KEY = 'TFMODEL_SAVED_MODEL_DIR'
    _PREPROCESSOR_KEY = 'TFMODEL_PREPROCESSOR_DIR'

    def __init__(self, app=None):
        self.app = app
        if self.app is not None:
            self.init_app(self.app)

    def init_app(self, app):
        self._init_saved_model(app)
        self._init_preprocessor(app)

    def _init_saved_model(self, app):
        app.config.setdefault(self._SAVED_MODEL_KEY, None)

    def _init_preprocessor(self, app):
        app.config.setdefault(self._PREPROCESSOR_KEY, None)

    def _create_predictor(self):
        path = current_app.config.setdefault(self._SAVED_MODEL_KEY, None)
        file = _get_last_file(path)
        predictor = tf.contrib.predictor.from_saved_model(file)
        return predictor

    def _create_preprocessor(self):
        path = current_app.config.setdefault(self._PREPROCESSOR_KEY, None)
        file = _get_last_file(path)
        with open(file, 'rb') as file:
            preprocessor = cloudpickle.load(file)
        return preprocessor

    @property
    def predictor(self):
        ctx = _app_ctx_stack.top
        if ctx is not None:
            try:
                return ctx.tfmodel_predictor
            except AttributeError:
                ctx.tfmodel_predictor = self._create_predictor()
                return ctx.tfmodel_predictor

    @property
    def preprocessor(self):
        ctx = _app_ctx_stack.top
        if ctx is not None:
            try:
                return ctx.tfmodel_preprocessor
            except AttributeError:
                ctx.tfmodel_preprocessor = self._create_preprocessor()
                return ctx.tfmodel_preprocessor

