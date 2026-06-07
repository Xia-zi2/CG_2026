import { SelectionOp } from '../edit-ops';

type SelectionModifierEvent = Pick<MouseEvent, 'shiftKey' | 'ctrlKey' | 'metaKey' | 'altKey'>;

const selectionOpFromPointerEvent = (event: SelectionModifierEvent): SelectionOp => {
    if (event.ctrlKey || event.metaKey) {
        return 'remove';
    }
    if (event.altKey) {
        return 'intersect';
    }
    if (event.shiftKey) {
        return 'add';
    }
    return 'set';
};

export { selectionOpFromPointerEvent };
