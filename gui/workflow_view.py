"""Live, compact visualization of the research workflow."""

from __future__ import annotations

import math
import time
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


class WorkflowProgressView(QWidget):
    """Show the parallel search lanes and live state of each workflow stage."""

    CANVAS_WIDTH = 1000
    CANVAS_HEIGHT = 590

    NODE_RECTS = {
        "understand_input": QRectF(40, 82, 250, 78),
        "planner": QRectF(375, 82, 250, 78),
        "search": QRectF(20, 235, 310, 100),
        "official_search": QRectF(670, 235, 310, 100),
        "evaluate": QRectF(380, 385, 240, 80),
        "board_update": QRectF(380, 500, 240, 70),
        "analyze": QRectF(40, 500, 250, 70),
        "additional_search": QRectF(710, 500, 250, 70),
    }
    NODE_LABELS = {
        "understand_input": "依頼内容を整理",
        "planner": "調査計画を作成",
        "search": "通常のWeb検索",
        "official_search": "公式情報を検索",
        "evaluate": "根拠と情報の不足を評価",
        "board_update": "確認事項を整理",
        "additional_search": "不足分を追加検索",
        "analyze": "レポートを作成",
    }
    NODE_HINTS = {
        "understand_input": "対象・目的・条件を確認",
        "planner": "検索の観点とクエリを準備",
        "search": "計画した観点を並行して検索",
        "official_search": "一次情報の候補を順に検索",
        "evaluate": "出典・本文・不足項目を確認",
        "board_update": "次の検索に使う情報を記録",
        "additional_search": "不足している観点を再調査",
        "analyze": "取得した根拠から回答を生成",
    }

    COLORS = {
        "ink": QColor("#172033"),
        "muted": QColor("#637083"),
        "line": QColor("#cbd5e1"),
        "blue": QColor("#2563eb"),
        "blue_fill": QColor("#eff6ff"),
        "green": QColor("#16804a"),
        "green_fill": QColor("#edf9f1"),
        "red": QColor("#be3e3e"),
        "red_fill": QColor("#fff1f0"),
        "gray": QColor("#94a3b8"),
        "gray_fill": QColor("#f3f5f8"),
        "card": QColor("#ffffff"),
        "canvas": QColor("#f7f9fc"),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(760, 500)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("調査ワークフローの進行状況")
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self.update)
        self.reset()

    def sizeHint(self) -> QSize:
        return QSize(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)

    def reset(self) -> None:
        self._elapsed_timer.stop()
        self._states = {name: "pending" for name in self.NODE_RECTS}
        self._active_counts = {name: 0 for name in self.NODE_RECTS}
        self._details: dict[str, set[str]] = {name: set() for name in self.NODE_RECTS}
        self._activity: dict[str, str] = {name: "" for name in self.NODE_RECTS}
        self._overall = "待機中"
        self._overall_detail = "調査を開始すると、検索レーンと各工程の状態がここに表示されます。"
        self._started_at: float | None = None
        self._elapsed_seconds = 0
        self.update()

    def begin_run(self) -> None:
        self.reset()
        self._overall = "調査中"
        self._overall_detail = "ワークフローを初期化しています。"
        self._started_at = time.monotonic()
        self._elapsed_timer.start()
        self.update()

    def update_progress(self, stage: str, status: str, detail: str = "") -> None:
        if stage == "workflow":
            self._elapsed_timer.stop()
            if self._started_at is not None:
                self._elapsed_seconds = int(time.monotonic() - self._started_at)
            if status == "completed":
                for name, state in self._states.items():
                    if state == "pending":
                        self._states[name] = "skipped"
                    self._active_counts[name] = 0
                    self._details[name].clear()
                self._overall = "調査完了"
                self._overall_detail = "レポートと参照ソースを確認できます。"
            elif status == "failed":
                for name, state in self._states.items():
                    if state == "pending":
                        self._states[name] = "skipped"
                    elif self._active_counts[name]:
                        self._states[name] = "failed"
                    self._active_counts[name] = 0
                    self._details[name].clear()
                self._overall = "調査に失敗しました"
                self._overall_detail = detail or "詳細実行ログを確認してください。"
            self.update()
            return

        if stage not in self._states:
            return
        if status == "started":
            self._active_counts[stage] += 1
            self._states[stage] = "active"
            if detail:
                self._details[stage].add(detail)
                self._activity[stage] = detail
            self._overall = "調査中"
            self._overall_detail = self._activity_text(stage, detail)
        elif status == "progress":
            if detail:
                self._activity[stage] = detail
            self._overall = "調査中"
            self._overall_detail = self._activity_text(stage, detail or self._activity[stage])
        elif status == "completed":
            self._active_counts[stage] = max(0, self._active_counts[stage] - 1)
            if self._active_counts[stage] == 0:
                self._states[stage] = "completed"
                self._details[stage].clear()
            elif detail:
                self._details[stage].discard(detail)
        elif status == "failed":
            self._active_counts[stage] = max(0, self._active_counts[stage] - 1)
            self._states[stage] = "failed" if self._active_counts[stage] == 0 else "active"
            self._overall = "調査中"
            self._overall_detail = detail or f"{self.NODE_LABELS[stage]}で問題が発生しました。"
            self._activity[stage] = self._overall_detail
        self.update()

    def _activity_text(self, stage: str, detail: str) -> str:
        title = self.NODE_LABELS.get(stage, stage)
        return f"{title} · {detail}" if detail else f"{title}を進めています。"

    def _elapsed_text(self) -> str:
        seconds = self._elapsed_seconds
        if self._started_at is not None and self._elapsed_timer.isActive():
            seconds = int(time.monotonic() - self._started_at)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"

    def _node_state_label(self, name: str) -> str:
        state = self._states[name]
        if state == "active":
            count = self._active_counts[name]
            if name == "search" and count > 1:
                return f"実行中 · {count}件を並行処理"
            return "実行中"
        return {
            "pending": "待機中",
            "completed": "完了",
            "skipped": "対象外",
            "failed": "失敗",
        }.get(state, state)

    def _node_detail(self, name: str) -> str:
        state = self._states[name]
        if state == "active":
            current = self._activity[name]
            if current:
                return current
            details = sorted(self._details[name])
            if details:
                return details[0] if len(details) == 1 else f"{len(details)}件を並行処理中"
        elif self._activity[name] and state in {"completed", "failed"}:
            return self._activity[name]
        return self.NODE_HINTS[name]

    def _stage_counts(self) -> tuple[int, int]:
        complete = sum(state in {"completed", "skipped"} for state in self._states.values())
        return complete, len(self._states)

    def _draw_arrow_head(self, painter: QPainter, start: QPointF, end: QPointF, color: QColor) -> None:
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        size = 8
        left = QPointF(
            end.x() - size * math.cos(angle - math.pi / 6),
            end.y() - size * math.sin(angle - math.pi / 6),
        )
        right = QPointF(
            end.x() - size * math.cos(angle + math.pi / 6),
            end.y() - size * math.sin(angle + math.pi / 6),
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([end, left, right]))

    def _edge_color(self, source: str, target: str) -> QColor:
        if self._states[target] == "active":
            return self.COLORS["blue"]
        if self._states[source] == "completed" and self._states[target] in {"completed", "skipped"}:
            return self.COLORS["green"]
        return self.COLORS["line"]

    def _draw_connector(
        self,
        painter: QPainter,
        points: list[QPointF],
        source: str,
        target: str,
        *,
        label: str = "",
        label_rect: QRectF | None = None,
    ) -> None:
        color = self._edge_color(source, target)
        path = QPainterPath(points[0])
        for point in points[1:]:
            path.lineTo(point)
        painter.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        self._draw_arrow_head(painter, points[-2], points[-1], color)
        if label:
            painter.setPen(self.COLORS["muted"])
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.drawText(label_rect or QRectF(points[-1].x() - 50, points[-1].y() - 20, 100, 18), Qt.AlignmentFlag.AlignCenter, label)

    def _draw_retry_loop(self, painter: QPainter) -> None:
        color = self._edge_color("additional_search", "evaluate")
        path = QPainterPath(QPointF(835, 500))
        path.cubicTo(QPointF(900, 465), QPointF(955, 465), QPointF(955, 430))
        path.cubicTo(QPointF(955, 395), QPointF(785, 425), QPointF(620, 425))
        painter.setPen(QPen(color, 2.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        self._draw_arrow_head(painter, QPointF(785, 425), QPointF(620, 425), color)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        painter.setPen(self.COLORS["muted"])
        painter.drawText(QRectF(800, 418, 105, 18), Qt.AlignmentFlag.AlignCenter, "再評価")

    def _draw_node(self, painter: QPainter, name: str, rect: QRectF) -> None:
        state = self._states[name]
        palette = {
            "pending": (self.COLORS["card"], self.COLORS["line"], self.COLORS["gray"]),
            "active": (self.COLORS["blue_fill"], self.COLORS["blue"], self.COLORS["blue"]),
            "completed": (self.COLORS["green_fill"], self.COLORS["green"], self.COLORS["green"]),
            "skipped": (self.COLORS["gray_fill"], self.COLORS["line"], self.COLORS["gray"]),
            "failed": (self.COLORS["red_fill"], self.COLORS["red"], self.COLORS["red"]),
        }
        fill, border, accent = palette[state]
        painter.setBrush(fill)
        painter.setPen(QPen(border, 2.4 if state == "active" else 1.3))
        painter.drawRoundedRect(rect, 12, 12)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(accent)
        painter.drawRoundedRect(QRectF(rect.left(), rect.top() + 10, 4, rect.height() - 20), 2, 2)

        painter.setPen(self.COLORS["ink"])
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        title_rect = QRectF(rect.x() + 16, rect.y() + 10, rect.width() - 112, 22)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.NODE_LABELS[name])

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
        status_rect = QRectF(rect.right() - 92, rect.y() + 10, 80, 22)
        painter.setPen(accent)
        painter.drawText(status_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, self._node_state_label(name))

        painter.setPen(self.COLORS["muted"])
        painter.setFont(QFont("Segoe UI", 8))
        hint_rect = QRectF(rect.x() + 16, rect.y() + 35, rect.width() - 28, 18)
        painter.drawText(hint_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self.NODE_HINTS[name])

        detail = self._node_detail(name)
        detail_rect = QRectF(rect.x() + 16, rect.y() + rect.height() - 27, rect.width() - 28, 18)
        painter.setPen(accent if state in {"active", "failed"} else self.COLORS["muted"])
        metrics = QFontMetrics(painter.font())
        painter.drawText(
            detail_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            metrics.elidedText(detail, Qt.TextElideMode.ElideRight, int(detail_rect.width())),
        )

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.COLORS["canvas"])
        scale = min(self.width() / self.CANVAS_WIDTH, self.height() / self.CANVAS_HEIGHT)
        painter.translate(
            (self.width() - self.CANVAS_WIDTH * scale) / 2,
            (self.height() - self.CANVAS_HEIGHT * scale) / 2,
        )
        painter.scale(scale, scale)

        painter.setPen(self.COLORS["ink"])
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        painter.drawText(QRectF(28, 15, 500, 26), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "調査フロー")

        completed, total = self._stage_counts()
        badge_color = self.COLORS["blue"] if self._overall == "調査中" else self.COLORS["green"] if self._overall == "調査完了" else self.COLORS["red"] if self._overall == "調査に失敗しました" else self.COLORS["muted"]
        painter.setPen(badge_color)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.DemiBold))
        painter.drawText(
            QRectF(690, 16, 170, 22),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{self._overall}  ·  {completed}/{total}工程",
        )
        painter.setPen(self.COLORS["muted"])
        painter.setFont(QFont("Segoe UI", 9))
        painter.drawText(
            QRectF(865, 16, 105, 22),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            self._elapsed_text(),
        )
        painter.setPen(QPen(QColor("#e3e8ef"), 1))
        painter.drawLine(QPointF(28, 48), QPointF(972, 48))
        painter.setPen(self.COLORS["muted"])
        painter.setFont(QFont("Segoe UI", 9))
        detail_metrics = QFontMetrics(painter.font())
        header_detail = detail_metrics.elidedText(self._overall_detail, Qt.TextElideMode.ElideRight, 930)
        painter.drawText(QRectF(28, 55, 944, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, header_detail)

        input_rect = self.NODE_RECTS["understand_input"]
        planner_rect = self.NODE_RECTS["planner"]
        search_rect = self.NODE_RECTS["search"]
        official_rect = self.NODE_RECTS["official_search"]
        evaluate_rect = self.NODE_RECTS["evaluate"]
        board_rect = self.NODE_RECTS["board_update"]
        analyze_rect = self.NODE_RECTS["analyze"]
        additional_rect = self.NODE_RECTS["additional_search"]

        self._draw_connector(
            painter,
            [QPointF(input_rect.right(), input_rect.center().y()), QPointF(planner_rect.left(), planner_rect.center().y())],
            "understand_input", "planner",
        )
        self._draw_connector(
            painter,
            [QPointF(planner_rect.center().x(), planner_rect.bottom()), QPointF(planner_rect.center().x(), 190), QPointF(search_rect.center().x(), 190), QPointF(search_rect.center().x(), search_rect.top())],
            "planner", "search",
        )
        self._draw_connector(
            painter,
            [QPointF(planner_rect.center().x(), planner_rect.bottom()), QPointF(planner_rect.center().x(), 190), QPointF(official_rect.center().x(), 190), QPointF(official_rect.center().x(), official_rect.top())],
            "planner", "official_search",
        )
        self._draw_connector(
            painter,
            [QPointF(search_rect.center().x(), search_rect.bottom()), QPointF(search_rect.center().x(), 360), QPointF(evaluate_rect.left() + 50, 360), QPointF(evaluate_rect.left() + 50, evaluate_rect.top())],
            "search", "evaluate",
        )
        self._draw_connector(
            painter,
            [QPointF(official_rect.center().x(), official_rect.bottom()), QPointF(official_rect.center().x(), 360), QPointF(evaluate_rect.right() - 50, 360), QPointF(evaluate_rect.right() - 50, evaluate_rect.top())],
            "official_search", "evaluate",
        )
        self._draw_connector(
            painter,
            [QPointF(evaluate_rect.center().x(), evaluate_rect.bottom()), QPointF(board_rect.center().x(), board_rect.top())],
            "evaluate", "board_update",
        )
        self._draw_connector(
            painter,
            [QPointF(board_rect.left(), board_rect.center().y()), QPointF(analyze_rect.right(), analyze_rect.center().y())],
            "board_update", "analyze",
            label="回答可能",
            label_rect=QRectF(285, 515, 95, 20),
        )
        self._draw_connector(
            painter,
            [QPointF(board_rect.right(), board_rect.center().y()), QPointF(additional_rect.left(), additional_rect.center().y())],
            "board_update", "additional_search",
            label="不足あり",
            label_rect=QRectF(620, 515, 90, 20),
        )
        self._draw_retry_loop(painter)

        for name, rect in self.NODE_RECTS.items():
            self._draw_node(painter, name, rect)
        painter.end()
