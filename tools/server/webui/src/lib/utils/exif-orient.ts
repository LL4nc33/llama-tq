import { MimeTypeImage } from '$lib/enums';

/**
 * MIME types that may carry EXIF orientation metadata (camera/phone photos).
 */
const EXIF_ORIENTABLE_MIME_TYPES = new Set<string>([MimeTypeImage.JPEG, MimeTypeImage.JPG]);

/**
 * Check whether a MIME type can carry EXIF orientation that needs correcting.
 * @param mimeType - The MIME type to check
 * @returns True if the image may need EXIF-orientation handling
 */
export function mayHaveExifOrientation(mimeType: string): boolean {
	return EXIF_ORIENTABLE_MIME_TYPES.has(mimeType);
}

/**
 * Bake EXIF orientation into the pixels and return a clean data URL.
 *
 * Phone photos store their rotation in an EXIF orientation tag rather than in
 * the pixels, so portrait shots arrive sideways at a model that ignores EXIF.
 * `createImageBitmap(..., { imageOrientation: 'from-image' })` lets the browser
 * apply the rotation natively; drawing the result onto a canvas re-encodes the
 * image with the rotation already applied. The canvas pass also strips all EXIF
 * metadata (incl. GPS) as a side effect — a privacy/DSGVO win. (upstream #24196)
 *
 * Falls back to the raw data URL when the browser lacks `createImageBitmap` or
 * the decode fails, so uploads never break.
 *
 * @param file - The original image File
 * @param fallbackDataURL - Pre-read data URL to return on failure
 * @returns Promise resolving to an orientation-corrected data URL
 */
export async function applyExifOrientation(file: File, fallbackDataURL: string): Promise<string> {
	if (typeof createImageBitmap !== 'function' || typeof document === 'undefined') {
		return fallbackDataURL;
	}

	let bitmap: ImageBitmap | undefined;
	try {
		bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });

		const canvas = document.createElement('canvas');
		canvas.width = bitmap.width;
		canvas.height = bitmap.height;

		const ctx = canvas.getContext('2d');
		if (!ctx) return fallbackDataURL;

		ctx.drawImage(bitmap, 0, 0);

		// Re-encode as JPEG at high quality; this drops the original EXIF block.
		return canvas.toDataURL(MimeTypeImage.JPEG, 0.92);
	} catch (err) {
		console.error('Failed to apply EXIF orientation, using original:', err);
		return fallbackDataURL;
	} finally {
		bitmap?.close();
	}
}
