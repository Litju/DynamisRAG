"""Relational records for the closed evidence-v1 contracts."""

from uuid import UUID

from sqlalchemy import ForeignKey, ForeignKeyConstraint, Integer, Text, UniqueConstraint, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from dynamisrag.db.models import Base


class EvidenceSetRecord(Base):
    __tablename__ = "evidence_set"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class ClaimRecord(Base):
    __tablename__ = "claim"
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class EvidenceMembershipRecord(Base):
    __tablename__ = "evidence_membership"
    __table_args__ = (UniqueConstraint("set_key", "ordinal", name="uq_evidence_member_order"),)
    set_key: Mapped[str] = mapped_column(ForeignKey("evidence_set.key"), primary_key=True)
    passage_key: Mapped[str] = mapped_column(Text, primary_key=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    passage_id: Mapped[UUID] = mapped_column(ForeignKey("passage.id"), nullable=False)
    version_id: Mapped[UUID] = mapped_column(ForeignKey("document_version.id"), nullable=False)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)


class EvidenceLinkRecord(Base):
    __tablename__ = "evidence_link"
    __table_args__ = (
        ForeignKeyConstraint(
            ["set_key", "passage_key"],
            ["evidence_membership.set_key", "evidence_membership.passage_key"],
            name="fk_evidence_link_member",
        ),
    )
    key: Mapped[str] = mapped_column(Text, primary_key=True)
    claim_key: Mapped[str] = mapped_column(ForeignKey("claim.key"), nullable=False)
    set_key: Mapped[str] = mapped_column(ForeignKey("evidence_set.key"), nullable=False)
    passage_key: Mapped[str] = mapped_column(Text, nullable=False)
    span_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("passage_source_span.id"), nullable=False
    )
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
