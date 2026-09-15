import React from 'react';
import { Check, RotateCcw, Loader2 } from 'lucide-react';

/**
 * Universal InspectorActionBar component
 * Provides standard transactional staging action bar across feature inspectors
 * (Captions, Effects, Filters, Audio).
 *
 * Props:
 * - isDirty: boolean - whether there are unapplied draft mutations
 * - onApply: () => void - applies staged changes to master timeline / clip
 * - onCancel: () => void - discards staged changes and restores snapshot
 * - isApplying: boolean - loading state while applying/saving
 * - applyLabel: string - custom label for apply button (default: "Apply Changes")
 * - cancelLabel: string - custom label for cancel button (default: "Cancel")
 * - className: string - optional custom classes
 * - description: string - optional sub-hint
 */
export default function InspectorActionBar({
    isDirty = false,
    onApply,
    onCancel,
    isApplying = false,
    applyLabel = 'Apply Changes',
    cancelLabel = 'Cancel',
    className = '',
    description = '',
}) {
    return (
        <div className={`relative z-20 pt-3 mt-3 border-t border-rule flex flex-col gap-2 ${className}`}>
            <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5">
                    <span
                        className={`w-2 h-2 rounded-full transition-colors ${
                            isDirty ? 'bg-warn animate-pulse' : 'bg-ok/60'
                        }`}
                    />
                    <span className="text-[11px] font-mono lowercase tracking-wide text-muted">
                        {isDirty ? 'staged changes pending' : 'saved to timeline'}
                    </span>
                </div>
                {description && (
                    <span className="text-[10px] text-muted truncate max-w-[160px]">
                        {description}
                    </span>
                )}
            </div>

            <div className="flex items-center gap-2">
                <button
                    type="button"
                    onClick={(e) => {
                        e.stopPropagation();
                        onCancel && onCancel();
                    }}
                    disabled={isApplying}
                    className="btn-ghost py-1.5 px-3 text-xs flex items-center gap-1.5 hover:text-warn transition-colors pointer-events-auto cursor-pointer"
                    title="Discard staged changes and restore previous settings"
                >
                    <RotateCcw size={13} />
                    <span>{cancelLabel}</span>
                </button>

                <button
                    type="button"
                    onClick={(e) => {
                        e.stopPropagation();
                        onApply && onApply();
                    }}
                    disabled={isApplying}
                    className={`btn-primary flex-1 py-1.5 px-3 text-xs flex items-center justify-center gap-1.5 transition-all pointer-events-auto cursor-pointer ${
                        isDirty
                            ? 'ring-1 ring-brass shadow-sm opacity-100'
                            : 'opacity-80 hover:opacity-100'
                    }`}
                    title={isDirty ? 'Commit staged changes to timeline' : 'Commit current settings to timeline'}
                >
                    {isApplying ? (
                        <>
                            <Loader2 size={13} className="animate-spin text-brassink" />
                            <span>Applying…</span>
                        </>
                    ) : (
                        <>
                            <Check size={13} />
                            <span>{applyLabel}</span>
                        </>
                    )}
                </button>
            </div>
        </div>
    );
}
