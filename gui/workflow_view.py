"""Live drawing of the research workflow for the GUI."""

from __future__ import annotations

import math
from typing import Any

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QSizePolicy, QWidget


class WorkflowProgressView(QWidget):
    """Show each workflow stage and update its state as graph nodes run."""

    CANVAS_WIDTH = 900
    CANVAS_HEIGHT = 650

    NODE_RECTS = {
        "planner": QRectF(325, 55, 250, 68),
        "search": QRectF(65, 175, 250, 68),
        "official_search": QRectF(585, 175, 250, 68),
        "evaluate": QRectF(325, 285, 250, 68),
        "board_update": QRectF(325, 375, 250, 68),
        "additional_search": QRectF(585, 375, 250, 68),
        "analyze": QRectF(325, 480, 250, 68),
        "complete": QRectF(350, 575, 200, 58),
    }
    NODE_LABELS = {
        "planner": "調査計画を作る",
        "search": "通常のWeb検索",
        "official_search": "公式情報を探す",
        "evaluate": "情報の十分さを確認",
        "board_update": "調査ボードを更新",
        "additional_search": "不足分を追加検索",
        "analyze": "レポートを作る",
        "complete": "調査完了",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(650, 470)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName("調査ワークフローの進行状況")
        self.reset()

    def sizeHint(self) -> QSize:
        return QSize(self.CANVAS_WIDTH, self.CANVAS_HEIGHT)

    def reset(self) -> None:
        self._states = {name: "pending" for name in self.NODE_RECTS if name != "complete"}
        self._active_counts = {name: 0 for name in self.NODE_RECTS if name != "complete"}
        self._details: dict[str, set[str]] = {name: set() for name in self._states}
        self._overall = "待機中"
        self._overall_detail = "調査を開始すると、各段階がここでリアルタイムに更新されます。"
        self.update()

    def begin_run(self) -> None:
        self.reset()
        self._overall = "調査中"
        self._overall_detail = "ワークフローを初期化しています。"
        self.update()

    def update_progress(self, stage: str, status: str, detail: str = "") -> None:
        if stage == "workflow":
            if status == "completed":
                for name, state in self._states.items():
                    if state == "pending":
                        self._states[name] = "skipped"
                    self._active_counts[name] = 0
                self._overall = "調査完了"
                self._overall_detail = "レポートと参照ソースを確認できます。"
            elif status == "failed":
                for name, state in self._states.items():
                    if state == "pending":
                        self._states[name] = "skipped"
                    elif self._active_counts[name]:
                        self._states[name] = "failed"
                    self._active_counts[name] = 0
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
            self._overall = "調査中"
            self._overall_detail = self._activity_text(stage, detail)
        elif status == "completed":
            self._active_counts[stage] = max(0, self._active_counts[stage] - 1)
            if detail:
                self._details[stage].discard(detail)
            if self._active_counts[stage] == 0:
                self._states[stage] = "completed"
            else:
                self._states[stage] = "active"
        elif status == "failed":
            self._active_counts[stage] = max(0, self._active_counts[stage] - 1)
            self._states[stage] = "failed" if self._active_counts[stage] == 0 else "active"
            self._overall = "調査中"
            self._overall_detail = detail or f"{self.NODE_LABELS[stage]}で問題が発生しました。"
        self.update()

    def _activity_text(self, stage: str, detail: str) -> str:
        title = self.NODE_LABELS.get(stage, stage)
        if stage in ("search", "official_search", "additional_search") and detail:
            return f"{title}: {detail}"
        return f"{title}を進めています。"

    def _draw_arrow(
        self,
        painter: QPainter,
        start: QPointF,
        end: QPointF,
        *,
        label: str = "",
        label_position: QPointF | None = None,
        color: QColor | None = None,
    ) -> None:
        color = color or QColor("#8994a3")
        painter.setPen(QPen(color, 2))
        painter.drawLine(start, end)
        angle = math.atan2(end.y() - start.y(), end.x() - start.x())
        arrow_length = 9
        left = QPointF(
            end.x() - arrow_length * math.cos(angle - math.pi / 6),
            end.y() - arrow_length * math.sin(angle - math.pi / 6),
        )
        right = QPointF(
            end.x() - arrow_length * math.cos(angle + math.pi / 6),
            end.y() - arrow_length * math.sin(angle + math.pi / 6),
        )
        painter.setBrush(color)
        painter.drawPolygon(QPolygonF([end, left, right]))
        if label:
            position = label_position or QPointF((start.x() + end.x()) / 2, (start.y() + end.y()) / 2)
            painter.setPen(QColor("#586273"))
            painter.drawText(QRectF(position.x() - 70, position.y() - 12, 140, 22), Qt.AlignmentFlag.AlignCenter, label)

    def _draw_loop(self, painter: QPainter) -> None:
        path = QPainterPath(QPointF(710, 375))
        path.cubicTo(QPointF(865, 340), QPointF(865, 305), QPointF(575, 319))
        painter.setPen(QPen(QColor("#8994a3"), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        end = QPointF(575, 319)
        before = path.pointAtPercent(0.98)
        angle = math.atan2(end.y() - before.y(), end.x() - before.x())
        left = QPointF(end.x() - 9 * math.cos(angle - math.pi / 6), end.y() - 9 * math.sin(angle - math.pi / 6))
        right = QPointF(end.x() - 9 * math.cos(angle + math.pi / 6), end.y() - 9 * math.sin(angle + math.pi / 6))
        painter.setBrush(QColor("#8994a3"))
        painter.drawPolygon(QPolygonF([end, left, right]))
        painter.setPen(QColor("#586273"))
        painter.drawText(QRectF(740, 300, 110, 24), Qt.AlignmentFlag.AlignCenter, "再評価へ")

    def _draw_node(self, painter: QPainter, name: str, rect: QRectF) -> None:
        state = self._states.get(name, "pending")
        fills = {
            "pending": QColor("#f0f2f5"),
            "active": QColor("#fff1c7"),
            "completed": QColor("#dff3e5"),
            "skipped": QColor("#f4f4f4"),
            "failed": QColor("#ffe1e1"),
        }
        borders = {
            "pending": QColor("#c6ccd5"),
            "active": QColor("#d99a00"),
            "completed": QColor("#3b9a5b"),
            "skipped": QColor("#c6ccd5"),
            "failed": QColor("#cc4545"),
        }
        state_labels = {
            "pending": "これから",
            "active": "実行中",
            "completed": "完了",
            "skipped": "未実施",
            "failed": "失敗",
        }
        if name == "complete":
            state = "completed" if self._overall == "調査完了" else "failed" if self._overall == "調査に失敗しました" else "pending"
        painter.setBrush(fills[state])
        style = Qt.PenStyle.DashLine if state == "skipped" else Qt.PenStyle.SolidLine
        painter.setPen(QPen(borders[state], 3 if state == "active" else 2, style))
        painter.drawRoundedRect(rect, 12, 12)
        painter.setPen(QColor("#1f2937"))
        name_font = painter.font()
        name_font.setBold(True)
        name_font.setPointSize(10)
        painter.setFont(name_font)
        painter.drawText(QRectF(rect.x() + 10, rect.y() + 7, rect.width() - 20, 28), Qt.AlignmentFlag.AlignCenter, self.NODE_LABELS[name])
        subfont = painter.font()
        subfont.setBold(False)
        subfont.setPointSize(8)
        painter.setFont(subfont)
        if state == "active" and self._details[name]:
            details = list(self._details[name])
            detail_text = details[0] if len(details) == 1 else f"{len(details)}件を並行処理中"
            if len(detail_text) > 37:
                detail_text = detail_text[:36] + "…"
            status_text = f"実行中: {detail_text}"
        elif name == "complete" and state == "completed":
            status_text = "結果を確認できます"
        else:
            status_text = state_labels[state]
        painter.setPen(QColor("#4b5563"))
        painter.drawText(QRectF(rect.x() + 8, rect.y() + 37, rect.width() - 16, rect.height() - 40), Qt.AlignmentFlag.AlignCenter, status_text)

    def paintEvent(self, event: Any) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#ffffff"))
        scale = min(self.width() / self.CANVAS_WIDTH, self.height() / self.CANVAS_HEIGHT)
        painter.translate((self.width() - self.CANVAS_WIDTH * scale) / 2, (self.height() - self.CANVAS_HEIGHT * scale) / 2)
        painter.scale(scale, scale)
        painter.setPen(QColor("#202938"))
        title_font = painter.font()
        title_font.setBold(True)
        title_font.setPointSize(13)
        painter.setFont(title_font)
        painter.drawText(QRectF(24, 10, self.CANVAS_WIDTH - 48, 26), Qt.AlignmentFlag.AlignLeft, f"進行状況: {self._overall}")
        body_font = painter.font()
        body_font.setBold(False)
        body_font.setPointSize(9)
        painter.setFont(body_font)
        painter.setPen(QColor("#596273"))
        painter.drawText(QRectF(24, 35, self.CANVAS_WIDTH - 48, 20), Qt.AlignmentFlag.AlignLeft, self._overall_detail)

        planner = self.NODE_RECTS["planner"]
        search = self.NODE_RECTS["search"]
        official = self.NODE_RECTS["official_search"]
        evaluate = self.NODE_RECTS["evaluate"]
        board = self.NODE_RECTS["board_update"]
        additional = self.NODE_RECTS["additional_search"]
        analyze = self.NODE_RECTS["analyze"]
        complete = self.NODE_RECTS["complete"]
        self._draw_arrow(painter, QPointF(planner.center().x() - 30, planner.bottom()), QPointF(search.center().x(), search.top()))
        self._draw_arrow(painter, QPointF(planner.center().x() + 30, planner.bottom()), QPointF(official.center().x(), official.top()))
        self._draw_arrow(painter, QPointF(search.right(), search.center().y()), QPointF(evaluate.left(), evaluate.center().y() - 8))
        self._draw_arrow(painter, QPointF(official.left(), official.center().y()), QPointF(evaluate.right(), evaluate.center().y() - 8))
        self._draw_arrow(painter, QPointF(evaluate.center().x(), evaluate.bottom()), QPointF(board.center().x(), board.top()))
        self._draw_arrow(painter, QPointF(board.center().x(), board.bottom()), QPointF(analyze.center().x(), analyze.top()), label="追加調査しない", label_position=QPointF(450, 465))
        self._draw_arrow(painter, QPointF(board.right(), board.center().y()), QPointF(additional.left(), additional.center().y()), label="不足がある", label_position=QPointF(580, 355))
        self._draw_loop(painter)
        self._draw_arrow(painter, QPointF(analyze.center().x(), analyze.bottom()), QPointF(complete.center().x(), complete.top()))
        for name, rect in self.NODE_RECTS.items():
            self._draw_node(painter, name, rect)
        painter.end()
